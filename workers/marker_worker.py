# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["marker-pdf==2.1.0"]
# ///
"""Long-lived marker-pdf worker.

marker 2.x on Apple Silicon uses rf-detr layout on MPS plus the Surya OCR VLM
served by llama.cpp (Metal). Both stay loaded for the life of this process.
"""

import os
import re
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _proto  # noqa: E402

from marker.converters.pdf import PdfConverter  # noqa: E402
from marker.models import create_model_dict  # noqa: E402
from marker.output import text_from_rendered  # noqa: E402

_models = None
SEP = "-" * 48
PAGE_RE = re.compile(r"\n*\{(\d+)\}" + SEP + r"\n*")


def _get_models():
    global _models
    if _models is None:
        t0 = time.perf_counter()
        _models = create_model_dict()
        return _models, time.perf_counter() - t0
    return _models, 0.0


def h_load(msg):
    _, secs = _get_models()
    return {"load_seconds": secs, "loaded": ["marker"]}


def h_unload(msg):
    global _models
    if _models and "inference_manager" in _models:
        try:
            _models["inference_manager"].stop()
        except Exception:
            pass
    _models = None
    return {"loaded": []}


def h_status(msg):
    return {"loaded": ["marker"] if _models else []}


def h_convert(msg):
    models, load_secs = _get_models()
    config = {
        "output_format": "markdown",
        "paginate_output": True,
        "page_separator": SEP,
        "disable_image_extraction": True,
        "force_ocr": bool(msg.get("force_ocr")),
    }
    if msg.get("mode"):
        config["mode"] = msg["mode"]
    if msg.get("page_range") is not None:
        config["page_range"] = msg["page_range"]
    t0 = time.perf_counter()
    converter = PdfConverter(artifact_dict=models, config=config)
    rendered = converter(msg["path"])
    markdown, _, _ = text_from_rendered(rendered)
    secs = time.perf_counter() - t0

    # Split "{N}----" separators into per-page chunks.
    parts = PAGE_RE.split("\n" + markdown)
    pages = []
    if len(parts) > 1:
        for i in range(1, len(parts) - 1, 2):
            pages.append({"page": int(parts[i]), "markdown": parts[i + 1].strip()})
    else:
        pages.append({"page": (msg.get("page_range") or [0])[0], "markdown": markdown.strip()})
    return {"pages": pages, "seconds": secs, "load_seconds": load_secs, "mode": converter.mode}


if __name__ == "__main__":
    _proto.serve({"load": h_load, "unload": h_unload, "status": h_status, "convert": h_convert})
