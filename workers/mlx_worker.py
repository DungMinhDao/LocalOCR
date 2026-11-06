# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git",
#   "mlx-lm",  # imported by some models' bundled MLX code (e.g. TeleOCR)
#   "pillow",
# ]
# ///
"""Long-lived MLX vision-language-model OCR worker (Apple Silicon GPU).

Keeps recently used models in memory so only the first page pays the load cost.
"""

import os
import sys
import tempfile
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _proto  # noqa: E402  (must be imported before mlx/transformers)

import mlx.core as mx  # noqa: E402
from mlx_vlm import load, stream_generate  # noqa: E402
from mlx_vlm.prompt_utils import apply_chat_template  # noqa: E402
from PIL import Image  # noqa: E402

MAX_LOADED = int(os.environ.get("LOCALOCR_MLX_MAX_MODELS", "2"))
_models: "OrderedDict[str, tuple]" = OrderedDict()


def _get(model_id: str):
    """Return (model, processor, load_seconds) loading on demand, LRU-evicting."""
    if model_id in _models:
        _models.move_to_end(model_id)
        return (*_models[model_id], 0.0)
    while len(_models) >= MAX_LOADED:
        _models.popitem(last=False)
        mx.clear_cache()
    t0 = time.perf_counter()
    path, revision = _resolve(model_id)
    model, processor = load(path, revision=revision, trust_remote_code=True)
    _models[model_id] = (model, processor)
    return model, processor, time.perf_counter() - t0


def _resolve(model_id: str) -> tuple[str, str | None]:
    """'repo@commit' pins a reviewed revision; 'convert:repo:bits' quantises the official weights once into
    data/models/ (for models whose community MLX conversions are broken)."""
    if model_id.startswith("convert:"):
        _, repo, bits = model_id.split(":")
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models",
                           f"{repo.split('/')[-1]}-{bits}bit")
        if not os.path.exists(os.path.join(out, "config.json")):
            from mlx_vlm.convert import convert
            convert(hf_path=repo, mlx_path=out, quantize=True, q_bits=int(bits))
        return out, None
    repo, _, revision = model_id.partition("@")
    return repo, revision or None


def _looping(text: str) -> bool:
    """Detect degenerate repetition (common failure mode of small OCR VLMs)."""
    lines = [l.strip() for l in text.rstrip().splitlines()[-16:]]
    if len(lines) == 16 and len(set(lines)) == 1:
        return True
    if len(text) < 300:
        return False
    for n in (8, 16, 32, 64, 128):
        unit = text[-n:]
        if unit.strip() and text.endswith(unit * max(3, 240 // n)):
            return True
    return False


# -- two-stage mode: PP-DocLayoutV3 finds blocks (with reading order), the VLM reads each crop ----------
LAYOUT_REPO = os.environ.get("LOCALOCR_LAYOUT_MODEL", "HashNuke/pp-doclayout-v3-mlx")
SKIP_LABELS = {"header", "footer", "number", "image", "chart", "seal", "formula_number"}
_layout = None


def _layout_model():
    global _layout
    if _layout is None:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from mlx_vlm.utils import load_model
        _layout = load_model(Path(snapshot_download(LAYOUT_REPO)))
    return _layout


def _inside(a, b, frac=0.8) -> bool:
    """Is box a (y0,x0,y1,x1) mostly inside box b?"""
    iy = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return iy * ix / area >= frac


def _blocks(image: Image.Image) -> list[dict]:
    found = [b for b in _layout_model().detect(image) if b["label"] not in SKIP_LABELS]
    found.sort(key=lambda b: b["reading_order"])
    keep = []
    for b in found:  # inline formulas and nested boxes: the enclosing block already reads them
        if any(o is not b and _inside(b["bbox"], o["bbox"]) and (o["bbox"][2] - o["bbox"][0]) *
               (o["bbox"][3] - o["bbox"][1]) > (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
               for o in found):
            continue
        keep.append(b)
    return keep


def _format(label: str, text: str) -> str:
    t = text.strip()
    if label == "formula":
        t = t.strip("$").strip()
        t = t.removeprefix("\\[").removesuffix("\\]").strip()
        return f"$$\n{t}\n$$"
    if label == "doc_title":
        return "# " + t.lstrip("# ")
    if label == "paragraph_title":
        return "## " + t.lstrip("# ")
    return t


def _generate(model, processor, rid, image_path, prompt, kwargs, on_text):
    """One generation; returns (text, last chunk, first-token seconds, stop reason)."""
    t0 = time.perf_counter()
    first_token = None
    text, last, stopped = "", None, None
    prompt = apply_chat_template(processor, model.config, prompt, num_images=1)
    for i, chunk in enumerate(stream_generate(model, processor, prompt, [image_path], **kwargs)):
        if getattr(chunk, "is_draft", False):
            continue
        if first_token is None:
            first_token = time.perf_counter() - t0
        text += chunk.text
        last = chunk
        on_text(chunk.text)
        if rid in _proto.cancelled:
            stopped = "cancelled"
            break
        if i % 16 == 0 and _looping(text):
            stopped = "repetition"
            break
    return text, last, first_token, stopped


def h_ocr_layout(msg, model, processor, load_secs, image_path, kwargs):
    rid = msg["id"]
    prompts = msg["layout"]
    page = Image.open(image_path).convert("RGB")
    t0 = time.perf_counter()
    # No blocks (a single line, a small crop): read the whole image as one text block
    blocks = _blocks(page) or [{"bbox": [0, 0, 1000, 1000], "label": "text", "reading_order": 1}]
    layout_secs = time.perf_counter() - t0
    W, H = page.size
    max_side = int(msg.get("max_side") or 0)
    parts, ptoks, gtoks, gen_secs, stopped, first = [], 0, 0, 0.0, None, None
    for b in blocks:
        y0, x0, y1, x1 = b["bbox"]
        pad = 6
        box = (max(0, int(x0 / 1000 * W) - pad), max(0, int(y0 / 1000 * H) - pad),
               min(W, int(x1 / 1000 * W) + pad), min(H, int(y1 / 1000 * H) + pad))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        crop = page.crop(box)
        if max_side and max(crop.size) > max_side:
            crop.thumbnail((max_side, max_side), Image.LANCZOS)
        kind = "table" if b["label"] == "table" else "formula" if b["label"] == "formula" else "text"
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            crop.save(f.name)
            if parts:
                _proto.send(id=rid, type="token", text="\n\n")
            text, last, ft, stopped = _generate(model, processor, rid, f.name, prompts.get(kind) or prompts["text"],
                                                kwargs, lambda t: _proto.send(id=rid, type="token", text=t))
        first = first if first is not None else ft
        if text.strip():
            parts.append(_format(b["label"], text))
        if last is not None:
            ptoks += getattr(last, "prompt_tokens", 0)
            gtoks += getattr(last, "generation_tokens", 0)
            gen_secs += getattr(last, "generation_tokens", 0) / max(getattr(last, "generation_tps", 0) or 1e-9, 1e-9)
        if stopped == "cancelled":
            break
    secs = time.perf_counter() - t0
    return {
        "text": "\n\n".join(parts), "load_seconds": load_secs, "seconds": secs, "first_token_seconds": first,
        "prompt_tokens": ptoks, "generation_tokens": gtoks, "prompt_tps": 0.0,
        "generation_tps": gtoks / gen_secs if gen_secs else 0.0, "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "finish_reason": stopped if stopped == "cancelled" else "stop", "blocks": len(blocks),
        "layout_seconds": layout_secs,
    }


def h_load(msg):
    _, _, secs = _get(msg["model"])
    return {"load_seconds": secs, "loaded": list(_models)}


def h_unload(msg):
    if msg.get("model"):
        _models.pop(msg["model"], None)
    else:
        _models.clear()
    mx.clear_cache()
    return {"loaded": list(_models)}


def h_status(msg):
    return {"loaded": list(_models)}


def h_ocr(msg):
    rid = msg["id"]
    if rid in _proto.cancelled:
        return {"text": "", "seconds": 0.0, "load_seconds": 0.0, "finish_reason": "cancelled"}
    model, processor, load_secs = _get(msg["model"])
    image_path = msg["image"]
    max_side = int(msg.get("max_side") or 0)
    if msg.get("layout"):
        kw = {"max_tokens": int(msg.get("max_tokens", 4096)), "temperature": float(msg.get("temperature", 0.0))}
        return h_ocr_layout(msg, model, processor, load_secs, image_path, kw)
    tmp = None
    if max_side:
        img = Image.open(image_path)
        if max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.convert("RGB").save(tmp.name)
            image_path = tmp.name

    prompt = apply_chat_template(processor, model.config, msg.get("prompt", ""), num_images=1)
    t0 = time.perf_counter()
    first_token = None
    text, last = "", None
    stopped = None
    kwargs = {"max_tokens": int(msg.get("max_tokens", 4096)), "temperature": float(msg.get("temperature", 0.0))}
    if msg.get("repetition_penalty"):
        kwargs["repetition_penalty"] = float(msg["repetition_penalty"])
    try:
        for i, chunk in enumerate(stream_generate(model, processor, prompt, [image_path], **kwargs)):
            if getattr(chunk, "is_draft", False):
                continue
            if first_token is None:
                first_token = time.perf_counter() - t0
            text += chunk.text
            last = chunk
            _proto.send(id=rid, type="token", text=chunk.text)
            if rid in _proto.cancelled:
                stopped = "cancelled"
                break
            if i % 16 == 0 and _looping(text):
                stopped = "repetition"
                break
    finally:
        if tmp:
            os.unlink(tmp.name)
    return {
        "text": text,
        "load_seconds": load_secs,
        "seconds": time.perf_counter() - t0,
        "first_token_seconds": first_token,
        "prompt_tokens": getattr(last, "prompt_tokens", 0),
        "generation_tokens": getattr(last, "generation_tokens", 0),
        "prompt_tps": getattr(last, "prompt_tps", 0.0),
        "generation_tps": getattr(last, "generation_tps", 0.0),
        "peak_memory_gb": getattr(last, "peak_memory", 0.0),
        "finish_reason": stopped or getattr(last, "finish_reason", None),
    }


if __name__ == "__main__":
    _proto.serve({"load": h_load, "unload": h_unload, "status": h_status, "ocr": h_ocr})
