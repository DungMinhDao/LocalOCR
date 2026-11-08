"""OCR engine adapters. Each engine turns a document's pages into Markdown + text."""

import base64
import io
import json
import mimetypes
import os
import re
from html import escape as html_escape
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from . import docs
from .convert import tex_formatting_to_md
from .paths import BIN, NATIVE, ROOT
from .worker_client import WorkerClient

HF_HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"

mlx_worker = WorkerClient("mlx", "mlx_worker.py")
marker_worker = WorkerClient("marker", "marker_worker.py")
WORKERS = {"mlx": mlx_worker, "marker": marker_worker}


def hf_cached(repo: str) -> bool:
    if repo.startswith("convert:"):  # quantised locally by the worker (see workers/mlx_worker.py:_resolve)
        _, src, bits = repo.split(":")
        return (ROOT / "data" / "models" / f"{src.split('/')[-1]}-{bits}bit" / "config.json").exists()
    d = HF_HUB / ("models--" + repo.split("@")[0].replace("/", "--")) / "snapshots"
    return d.exists() and any(d.iterdir())


def hard_breaks(text: str) -> str:
    """Keep OCR line breaks when rendered as Markdown (paragraphs stay separated)."""
    return re.sub(r"(?<=\S)\n(?=\S)", "  \n", text.strip())


# ---------------------------------------------------------------------------
class Engine:
    id = ""
    name = ""
    group = ""
    description = ""
    speed = ""          # rough speed hint shown in UI
    tag = ""            # what it is good for, a few words under the name
    accepts = ("pdf", "image")
    worker: WorkerClient | None = None

    def options(self) -> list[dict]:
        return []

    def status(self) -> dict:
        return {"available": True, "note": ""}

    def default_opts(self) -> dict:
        return {o["key"]: o.get("default") for o in self.options()}

    featured = True     # False: listed under "More models"

    def describe(self) -> dict:
        return {"id": self.id, "name": self.name, "group": self.group, "description": self.description,
                "speed": self.speed, "tag": self.tag, "accepts": list(self.accepts), "options": self.options(),
                "featured": self.featured,
                "worker": self.worker.name if self.worker else None, **self.status()}

    def run(self, job) -> None:
        raise NotImplementedError

    def warmup(self) -> dict:
        return {}

    def uses_gpu(self, opts: dict) -> bool:
        """GPU jobs share one lane so their timings don't distort each other."""
        return self.worker is not None


DPI_OPT = {"key": "dpi", "label": "PDF render DPI", "type": "select", "default": 200,
           "options": [100, 150, 200, 300], "help": "Only used for PDFs"}


# ---------------------------------------------------------------------------
class AppleVision(Engine):
    id = "apple-vision"
    name = "Apple Vision"
    group = "OCR · Apple Neural Engine"
    description = ("macOS built-in OCR. 'document' mode (macOS 26+) detects paragraphs, lists and tables; "
                   "'text' mode is the classic line recognizer and supports language hints.")
    speed = "⚡ 0.4 s/page"
    tag = "plain text, handwriting"
    binary = BIN / "vision-ocr"

    def options(self):
        return [
            {"key": "mode", "label": "Mode", "type": "select", "default": "document", "options": ["document", "text"]},
            {"key": "langs", "label": "Languages (text mode)", "type": "text", "default": "",
             "placeholder": "auto, or e.g. en-US,vi-VT"},
            {"key": "fast", "label": "Fast recognizer (text mode)", "type": "bool", "default": False},
            DPI_OPT,
        ]

    def ensure_binary(self) -> None:
        src = NATIVE / "vision_ocr.swift"
        if self.binary.exists() and self.binary.stat().st_mtime >= src.stat().st_mtime:
            return
        subprocess.run(["swiftc", "-O", "-parse-as-library", str(src), "-o", str(self.binary)],
                       check=True, capture_output=True, text=True)

    def status(self):
        if self.binary.exists() or shutil.which("swiftc"):
            return {"available": True, "note": ""}
        return {"available": False, "note": "Needs Xcode command line tools (swiftc) to build the helper"}

    def run(self, job):
        self.ensure_binary()
        o = job.opts
        paths = {str(docs.page_image(job.file_id, p, int(o.get("dpi") or 200))): p for p in job.pages}
        cmd = [str(self.binary), "--mode", o.get("mode") or "document"]
        if o.get("fast"):
            cmd.append("--fast")
        if o.get("langs"):
            cmd += ["--langs", o["langs"]]
        proc = subprocess.Popen(cmd + list(paths), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        job.proc = proc
        for line in proc.stdout:
            d = json.loads(line)
            if d.get("error"):
                raise RuntimeError(d["error"])
            job.page_done(paths[d["path"]], markdown=d["markdown"], text=d["text"], seconds=d["seconds"])
        if proc.wait() and not job.cancelled:
            raise RuntimeError(proc.stderr.read()[-2000:])


# ---------------------------------------------------------------------------
class Tesseract(Engine):
    id = "tesseract"
    name = "Tesseract"
    group = "OCR · CPU"
    description = "Classic LSTM OCR. Pages run in parallel across CPU cores. Supports searchable-PDF export."
    speed = "⚡ 0.9 s/page"
    tag = "searchable PDF export"

    def langs(self) -> list[str]:
        try:
            out = subprocess.run(["tesseract", "--list-langs"], capture_output=True, text=True).stdout
            return [l.strip() for l in out.splitlines()[1:] if l.strip() and l.strip() != "osd"]
        except FileNotFoundError:
            return []

    def options(self):
        langs = self.langs() or ["eng"]
        return [
            {"key": "lang", "label": "Language(s)", "type": "text", "default": "eng",
             "placeholder": "+".join(langs[:3]), "help": "Installed: " + ", ".join(langs)
             + (" — `brew install tesseract-lang` for more" if len(langs) < 5 else "")},
            {"key": "psm", "label": "Page segmentation", "type": "select", "default": "3",
             "options": [["3", "3 · auto"], ["4", "4 · single column"], ["6", "6 · uniform block"],
                         ["11", "11 · sparse text"], ["1", "1 · auto + OSD"]]},
            {**DPI_OPT, "default": 300},
        ]

    def status(self):
        if shutil.which("tesseract"):
            return {"available": True, "note": ""}
        return {"available": False, "note": "brew install tesseract"}

    def run(self, job):
        o = job.opts
        dpi = int(o.get("dpi") or 300)
        env = {**os.environ, "OMP_THREAD_LIMIT": "1"}  # parallelism comes from running pages concurrently

        def one(p):
            if job.cancelled:
                return
            img = docs.page_image(job.file_id, p, dpi)
            t0 = time.perf_counter()
            r = subprocess.run(["tesseract", str(img), "stdout", "-l", o.get("lang") or "eng",
                                "--psm", str(o.get("psm") or 3), "--dpi", str(dpi)],
                               capture_output=True, text=True, env=env)
            if r.returncode:
                raise RuntimeError(r.stderr.strip()[-1000:])
            text = r.stdout.replace("\f", "").strip()
            job.page_done(p, markdown=hard_breaks(text), text=text, seconds=time.perf_counter() - t0)

        with ThreadPoolExecutor(max(1, min(len(job.pages), os.cpu_count() or 4))) as ex:
            for f in [ex.submit(one, p) for p in job.pages]:
                f.result()


# ---------------------------------------------------------------------------
class PdfText(Engine):
    id = "pdf-text"
    name = "PDF text layer"
    tag = "digital PDFs"
    group = "Converters (digital files · no OCR)"
    description = "No OCR — extracts the embedded text of digital PDFs instantly. Useless for scans."
    speed = "⚡ instant"
    accepts = ("pdf",)

    def run(self, job):
        for p in job.pages:
            t0 = time.perf_counter()
            text = docs.pdf_text(job.file_id, p).strip()
            job.page_done(p, markdown=hard_breaks(text), text=text, seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------------------
OVIS_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order and output the result as a single "
    "Markdown document. For charts or images, represent them using an HTML image tag: <img src=\"images/bbox_{left}_"
    "{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard "
    "Markdown. Preserve the original text without translation or paraphrasing."
)
XIAOMI_PROMPT = (
    "Extract all information from the main body of the document image and represent it in markdown format, ignoring "
    "headers and footers. Tables should be expressed in OTSL format, formulas in the document should be represented "
    "using LATEX format, and the parsing should be organized according to the reading order."
)
# "layout": block prompts for two-stage mode (PP-DocLayoutV3 finds blocks + reading order, the VLM reads each crop).
# "featured": shown in the main engine list; the rest sit under "More models".
MLX_PRESETS = [
    {"id": "paddleocr-vl", "tag": "all-rounder, tables", "name": "PaddleOCR-VL 1.6", "model": "mlx-community/PaddleOCR-VL-1.6-4bit", "size_gb": 0.7,
     "prompt": "OCR:", "featured": True, "mode": "layout",
     "layout": {"text": "OCR:", "table": "Table Recognition:", "formula": "Formula Recognition:"},
     "description": "Baidu PaddleOCR-VL 1.6 0.9B (Apache-2.0), 109 languages. Best all-rounder in our benchmark (82.9, fastest of "
                    "the accurate models); best tables of the fast models. Updated every few months.",
     "speed": "4.1 s/page"},
    {"id": "mineru-2.5-pro", "tag": "Vietnamese documents", "name": "MinerU2.5-Pro", "model": "OpenGryd/MinerU2.5-Pro-2605-1.2B-MLX-8bit",
     "size_gb": 1.9, "prompt": "\nText Recognition:", "featured": True, "mode": "layout", "layout_only": True,
     "layout": {"text": "\nText Recognition:", "table": "\nTable Recognition:", "formula": "\nFormula Recognition:"},
     "description": "OpenDataLab MinerU2.5-Pro 1.2B (Apache-2.0): block-by-block; strong on Vietnamese documents and "
                    "English tests alike; monthly releases.",
     "speed": "6.5 s/page"},
    {"id": "unlimited-ocr-2", "tag": "long documents, photos", "name": "Unlimited-OCR-2", "model": "mlx-community/Unlimited-OCR-2-4bit", "size_gb": 2.6,
     "prompt": "Multi page parsing.", "featured": True, "mode": "page",
     "description": "Baidu Unlimited-OCR-2 3.3B (MIT, Oct 2026): whole page; built for long multi-page input. Strong on "
                    "English tests, Vietnamese and photographed pages; skips handwriting.",
     "speed": "8.0 s/page"},
    {"id": "xiaomi-ocr-0", "tag": "English only, fast", "name": "Xiaomi-OCR-0", "model": "playstonex/Xiaomi-OCR-0-4bit-mlx", "size_gb": 0.65,
     "prompt": XIAOMI_PROMPT, "featured": True, "mode": "page",
     "description": "Xiaomi-OCR-0 0.9B (Apache-2.0, Sep 2026): whole page, fast; very good on English tests and tables "
                    "(OTSL, converted to HTML). Drops Vietnamese diacritics. Community MLX conversion.",
     "speed": "5.0 s/page"},
    {"id": "teleocr", "tag": "English tests, tables", "name": "TeleOCR", "model": "groxaxo/TeleOCR-oQ8-MLX@de937230aca3d18f965ab038015034f330dc54d0", "size_gb": 2.1,
     "prompt": "\nPlease output the text content from the image.", "featured": True, "mode": "layout", "layout_only": True,
     "layout": {"text": "\nPlease output the text content from the image.",
                "table": "\nThis is the image of a table. Please output the table in OTSL format.",
                "formula": "\nPlease write out the expression of the formula in the image using LaTeX format."},
     "description": "TeleOCR 1.2B (Apache-2.0, Aug 2026): block-by-block; the most accurate on English tests, textbooks "
                    "and tables. Weak on Vietnamese and handwriting (zh/en only). Community MLX conversion whose bundled "
                    "model code was reviewed and is pinned to that commit.",
     "speed": "7.7 s/page"},
    {"id": "lighton-ocr-3-1b", "tag": "Vietnamese, photos", "name": "LightOnOCR-3 1B", "model": "convert:lightonai/LightOnOCR-3-1B:8",
     "size_gb": 2.0, "prompt": "", "mode": "page",
     "description": "LightOn LightOnOCR-3 1B (Apache-2.0, Oct 2026): whole page; very good on Vietnamese, photographed "
                    "pages and formulas, at ~9 s/page. Official weights quantised to 8-bit locally on first use.",
     "speed": "8.8 s/page"},
    {"id": "glm-ocr", "tag": "formulas", "name": "GLM-OCR", "model": "mlx-community/GLM-OCR-8bit", "size_gb": 1.6,
     "prompt": "Text Recognition:", "mode": "layout",
     "layout": {"text": "Text Recognition:", "table": "Table Recognition:", "formula": "Formula Recognition:"},
     "description": "Zhipu GLM-OCR 0.9B (MIT): block-by-block; best formulas (67.3) and strong tables and printed English. "
                    "Weaker on Vietnamese (82.2).",
     "speed": "6.0 s/page"},
    {"id": "lighton-ocr-3", "tag": "best Vietnamese, slow", "name": "LightOnOCR-3 4B", "model": "convert:lightonai/LightOnOCR-3-4B:4", "size_gb": 9.1,
     "prompt": "", "mode": "page",
     "description": "LightOn LightOnOCR-3 4B (Apache-2.0, Oct 2026): the most accurate on Vietnamese (98.5) and near-best on "
                    "handwriting (83.3), at ~15 s/page. Official weights quantised to 4-bit locally on first use.",
     "speed": "15 s/page"},
    {"id": "ovisocr2", "tag": "English pages", "name": "OvisOCR2", "model": "mlx-community/OvisOCR2-8bit", "size_gb": 1.0,
     "prompt": OVIS_PROMPT, "mode": "page",
     "description": "ATH OvisOCR2 0.8B (Apache-2.0): whole page; tops OmniDocBench but only 77.2 in our benchmark "
                    "(8-bit MLX, 1280 px), with Chinese characters leaking into Vietnamese.",
     "speed": "5.4 s/page"},
    {"id": "dots-mocr", "tag": "clean pages", "name": "dots.mocr", "model": "mlx-community/dots.mocr-4bit", "size_gb": 3.3,
     "prompt": "Read this test page and output the content in accurate human reading order as Markdown.",
     "mode": "page",
     "description": "rednote dots.mocr 3B (MIT): whole page; 77.5 in our benchmark, weak on degraded scans (63.1).",
     "speed": "8.8 s/page"},
    {"id": "custom", "tag": "any mlx-vlm model", "name": "Custom MLX model", "model": "", "size_gb": None,
     "prompt": "Convert this page to Markdown.", "mode": "page",
     "description": "Any mlx-vlm compatible model id from Hugging Face (e.g. mlx-community/Qwen3-VL-8B-Instruct-4bit).",
     "speed": ""},
]


def otsl_to_html(otsl: str) -> str:
    """OTSL table tokens (<fcel> text, <ecel>, <lcel> merge-left, <ucel> merge-up, <xcel>, <nl>) → HTML table."""
    body = re.sub(r"</?otsl>", "", otsl)
    rows, row = [], []
    for tok, text in re.findall(r"<(fcel|ecel|lcel|ucel|xcel|ched|rhed|srow|nl)>([^<]*)", body):
        if tok == "nl":
            rows.append(row)
            row = []
        else:
            row.append((tok, text.strip()))
    if row:
        rows.append(row)
    out = []
    for r, cells in enumerate(rows):
        tds = []
        for c, (tok, text) in enumerate(cells):
            if tok in ("lcel", "ucel", "xcel"):
                continue
            span = 1
            while c + span < len(cells) and cells[c + span][0] in ("lcel", "xcel"):
                span += 1
            rspan = 1
            while r + rspan < len(rows) and c < len(rows[r + rspan]) and rows[r + rspan][c][0] in ("ucel", "xcel"):
                rspan += 1
            attrs = (f' colspan="{span}"' if span > 1 else "") + (f' rowspan="{rspan}"' if rspan > 1 else "")
            tds.append(f"<td{attrs}>{html_escape(text)}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>" + "".join(out) + "</table>"


_OTSL = re.compile(r"(?:<otsl>)?(?:<(?:fcel|ecel|lcel|ucel|xcel|ched|rhed|srow)>[^<]*|<nl>\s*)+(?:</otsl>)?")


def clean_vlm(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:markdown|md)?\s*\n(.*?)\n```\s*$", r"\1", t, flags=re.S)       # wrapping fence
    t = re.sub(r"^---\n(?:[a-z_]+:.*\n)+---\n?", "", t)                                  # YAML front matter (olmOCR-style)
    t = re.sub(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>.*?<\|/det\|>\n?", "", t, flags=re.S)  # DeepSeek-style grounding
    t = re.sub(r"<\|(?:/?ref|/?det|im_end|endoftext|end_of_text)\|>", "", t)
    if re.search(r"!\[\w+\]\(\d+,\d+,\d+,\d+\)", t):  # LightOnOCR-3 grounding markers
        t = re.sub(r"!\[(?:page_number|header|footer|page_header|page_footer)\]\(\d+,\d+,\d+,\d+\)[^\n]*(?:\n(?!\n)[^\n]*)*",
                   "", t)
        t = re.sub(r"!\[\w+\]\(\d+,\d+,\d+,\d+\)[ \t]*\n?", "", t)
    if "<BLOCK>" in t:  # Unlimited-OCR: <PAGE><BLOCK>label[x0, y0, x1, y1]<FORK>content
        blocks = []
        for b in t.replace("<PAGE>", "").split("<BLOCK>"):
            label, _, body = b.partition("<FORK>")
            label = re.sub(r"\[.*", "", label).strip()
            if not body.strip() or label in ("page_number", "header", "footer", "page_header", "page_footer"):
                continue
            blocks.append(("# " if label == "title" else "") + body.strip())
        t = "\n\n".join(blocks)
    if "cel>" in t:
        t = _OTSL.sub(lambda m: "\n" + otsl_to_html(m.group(0)) + "\n", t)
    return tex_formatting_to_md(trim_repetition(t)).strip()


def trim_repetition(t: str, keep: int = 2) -> str:
    """Collapse a run of identical lines at the end (a stopped generation loop)."""
    lines = t.rstrip().split("\n")
    run = 1
    while run < len(lines) and lines[-1 - run].strip() == lines[-1].strip():
        run += 1
    if run > 4:
        lines = lines[: len(lines) - run + keep] + ["…"]
    return "\n".join(lines)


class MlxVlm(Engine):
    group = "OCR · vision-language models (MLX GPU)"
    worker = mlx_worker

    def __init__(self, preset: dict):
        self.preset = preset
        self.id = "mlx:" + preset["id"]
        self.name = preset["name"]
        self.description = preset["description"]
        self.speed = preset["speed"]
        self.tag = preset.get("tag", "")
        self.featured = bool(preset.get("featured"))

    def options(self):
        p = self.preset
        opts = []
        if p["id"] == "custom":
            opts.append({"key": "model", "label": "Model id", "type": "text", "default": "",
                         "placeholder": "mlx-community/…"})
        if p.get("layout") and not p.get("layout_only"):
            opts.append({"key": "mode", "label": "Mode", "type": "select", "default": p["mode"],
                         "options": [["layout", "Layout blocks (PP-DocLayoutV3 + crops)"], ["page", "Whole page"]],
                         "help": "Layout mode reads each detected block with a task prompt, as the model was "
                                 "trained; skips headers, footers, page numbers and pictures"})
        opts += [
            {"key": "max_side", "label": "Max image side (px)", "type": "select", "default": 1280,
             "options": [[0, "original"], [960, "960 · fastest"], [1280, "1280 · fast"], [1600, "1600"], [2048, "2048"]],
             "help": "Downscaling cuts vision tokens: the biggest speed lever for VLMs"},
            {"key": "max_tokens", "label": "Max tokens / page", "type": "number", "default": 4096},
            {"key": "prompt", "label": "Prompt", "type": "textarea", "default": p["prompt"]},
            {**DPI_OPT, "default": 150},
        ]
        return opts

    def status(self):
        model = self.preset["model"]
        if not model:
            return {"available": True, "note": "", "cached": None}
        cached = hf_cached(model)
        size = self.preset.get("size_gb")
        note = "" if cached else f"First run downloads {size} GB" if size else "First run downloads the model"
        return {"available": True, "note": note, "cached": cached, "model": model,
                "loaded": model in mlx_worker.loaded}

    def model_id(self, opts) -> str:
        return (opts.get("model") or "").strip() or self.preset["model"]

    def warmup(self):
        return mlx_worker.request("load", model=self.model_id({}))

    def run(self, job):
        o = job.opts
        model = self.model_id(o)
        if not model:
            raise ValueError("Enter a model id for the custom MLX engine")
        prompt = o.get("prompt")
        if prompt is None:
            prompt = self.preset["prompt"]
        for p in job.pages:
            if job.cancelled:
                return
            img = docs.page_image(job.file_id, p, int(o.get("dpi") or 150))
            job.current_page = p
            job.partial = ""
            if model not in mlx_worker.loaded:
                job.stage = "loading model" + ("" if hf_cached(model) else " (downloading…)")
            else:
                job.stage = "generating"
            rid = f"{job.id}-{p}"
            job.cancel_hook = lambda rid=rid: mlx_worker.cancel(rid)

            def on_event(m):
                if m.get("type") == "token":
                    job.stage = "generating"
                    job.partial += m["text"]

            layout = self.preset.get("layout") if (o.get("mode") or self.preset["mode"]) == "layout" else None
            r = mlx_worker.request(
                "ocr", on_event=on_event, rid=rid, model=model, image=str(img), prompt=prompt, layout=layout,
                max_tokens=int(o.get("max_tokens") or 4096), max_side=int(o.get("max_side") or 0),
            )
            md = clean_vlm(r["text"])
            if r.get("load_seconds"):
                job.load_seconds += r["load_seconds"]
            stats = {k: r.get(k) for k in ("prompt_tokens", "generation_tokens", "prompt_tps", "generation_tps",
                                            "first_token_seconds", "peak_memory_gb", "finish_reason", "blocks",
                                            "layout_seconds")}
            job.page_done(p, markdown=md, text=None, seconds=r["seconds"], stats=stats)
            if r.get("finish_reason") == "cancelled":
                return


# ---------------------------------------------------------------------------
def _strip_images(md: str) -> str:
    """Image extraction is disabled, so marker's image links point nowhere."""
    return re.sub(r"\n*!\[[^\]]*\]\([^)]*\)\n*", "\n\n", md).strip()


class Marker(Engine):
    id = "marker"
    name = "marker-pdf"
    group = "OCR · layout pipeline"
    description = ("Datalab marker 2.1: layout detection (rf-detr on MPS), PDF text-layer reuse, tables, "
                   "and Surya OCR VLM via llama.cpp/Metal for scanned text.")
    speed = "14 s/page"
    tag = "badly captured pages, slow"
    worker = marker_worker

    def options(self):
        return [
            {"key": "mode", "label": "Mode", "type": "select", "default": "fast",
             "options": [["fast", "fast · OCR only where needed"], ["balanced", "balanced · VLM layout + full OCR"]]},
            {"key": "force_ocr", "label": "Force OCR (ignore PDF text layer)", "type": "bool", "default": False},
        ]

    def status(self):
        if not shutil.which("llama-server"):
            return {"available": True, "note": "Scanned pages need llama.cpp: brew install llama.cpp"}
        return {"available": True, "note": "" if marker_worker.loaded else "First run installs marker + models (~1–3 min)"}

    def warmup(self):
        return marker_worker.request("load")

    def run(self, job):
        meta = docs.get(job.file_id)
        o = job.opts
        job.stage = "converting" if marker_worker.loaded else "loading marker"
        job.cancel_hook = None
        payload = {"mode": o.get("mode") or "fast", "force_ocr": bool(o.get("force_ocr"))}
        if meta["kind"] == "pdf":
            r = marker_worker.request("convert", path=str(docs.source_path(job.file_id)),
                                      page_range=job.pages, **payload)
            job.load_seconds += r.get("load_seconds", 0)
            per = r["seconds"] / max(1, len(r["pages"]))
            by_page = {pg["page"]: _strip_images(pg["markdown"]) for pg in r["pages"]}
            for p in job.pages:
                job.page_done(p, markdown=by_page.get(p, ""), text=None, seconds=per)
        else:
            for p in job.pages:
                img = docs.page_image(job.file_id, p)
                r = marker_worker.request("convert", path=str(img), **payload)
                job.load_seconds += r.get("load_seconds", 0)
                job.page_done(p, markdown=_strip_images("\n\n".join(pg["markdown"] for pg in r["pages"])),
                              text=None, seconds=r["seconds"])


# ---------------------------------------------------------------------------
class _MlxChatShim:
    """Duck-typed OpenAI client (`client.chat.completions.create`) backed by the MLX worker,
    so MarkItDown and its OCR plugin can use a local, already-warm model."""

    MIN_SIDE = 120

    def __init__(self, job, model: str, max_side: int, max_tokens: int):
        self.job, self.model, self.max_side, self.max_tokens = job, model, max_side, max_tokens
        self.calls = self.skipped = 0
        self.stats: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model=None, messages=(), **_):
        job = self.job
        if job.cancelled:
            raise RuntimeError("cancelled")
        prompt, data_uri = "", None
        for part in messages[-1]["content"]:
            if part.get("type") == "text":
                prompt = part["text"]
            elif part.get("type") == "image_url":
                data_uri = part["image_url"]["url"]
        header, b64 = data_uri.split(",", 1)
        ext = mimetypes.guess_extension(header[5:].split(";")[0]) or ".png"
        raw = base64.b64decode(b64)
        try:
            with Image.open(io.BytesIO(raw)) as im:
                w, h = im.size
        except Exception:
            w = h = 0
        if min(w, h) < self.MIN_SIDE:  # logos, bullets, rules: not worth a VLM call
            self.skipped += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))])
        self.calls += 1
        job.stage = f"OCR of image #{self.calls}" + ("" if self.model in mlx_worker.loaded else " (loading model)")
        job.partial = ""
        rid = f"{job.id}-md{self.calls}"
        job.cancel_hook = lambda: mlx_worker.cancel(rid)

        def on_event(m):
            if m.get("type") == "token":
                job.partial += m["text"]

        with tempfile.NamedTemporaryFile(suffix=ext) as f:
            f.write(raw)
            f.flush()
            r = mlx_worker.request("ocr", on_event=on_event, rid=rid, model=self.model, image=f.name,
                                   prompt=prompt, max_side=self.max_side, max_tokens=self.max_tokens)
        job.load_seconds += r.get("load_seconds") or 0
        self.stats.append(r)
        text = clean_vlm(r["text"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


CONVERTERS = "Converters (digital files · no OCR)"
IMG_LINK = re.compile(r"\n*!\[[^\]]*\]\([^)]*\)\n*")


class MarkItDownEngine(Engine):
    id = "markitdown"
    name = "MarkItDown"
    tag = "Word, PowerPoint, Excel"
    group = CONVERTERS
    description = ("Microsoft MarkItDown 0.1.9: DOCX, PPTX, XLSX/XLS, Outlook MSG, HTML, EPUB, PDF text → lean "
                   "Markdown (the fewest tokens of all converters). Not OCR: optionally the markitdown-ocr plugin "
                   "sends embedded images / scanned pages to a local MLX model.")
    speed = "⚡ instant"
    accepts = ("pdf", "image", "office")
    worker = mlx_worker

    def options(self):
        models = [["", "none · text layer only"]] + [
            [p["model"], f"{p['name']}" + ("" if hf_cached(p["model"]) else f" · downloads {p['size_gb']} GB")]
            for p in MLX_PRESETS if p["model"]]
        return [
            {"key": "ocr_model", "label": "OCR images / scans with", "type": "select",
             "default": "", "options": models},
            {"key": "drop_images", "label": "Drop image links (saves tokens)", "type": "bool", "default": True},
            {"key": "prompt", "label": "OCR prompt", "type": "textarea", "default": "",
             "placeholder": "blank = the chosen model's own prompt"},
            {"key": "max_side", "label": "Max image side (px)", "type": "select", "default": 1280,
             "options": [[0, "original"], [960, "960"], [1280, "1280"], [1600, "1600"]]},
        ]

    def uses_gpu(self, opts):
        return bool(opts.get("ocr_model"))

    def run(self, job):
        from markitdown import MarkItDown

        o = job.opts
        meta = docs.get(job.file_id)
        src = docs.source_path(job.file_id)
        model = o.get("ocr_model") or ""
        kwargs = {}
        shim = None
        if model:
            preset = next((p for p in MLX_PRESETS if p["model"] == model), None)
            prompt = (o.get("prompt") or "").strip() or (preset["prompt"] if preset else "") or \
                "Extract all text from this image as Markdown."
            shim = _MlxChatShim(job, model, int(o.get("max_side") or 0), 4096)
            kwargs = {"llm_client": shim, "llm_model": model, "llm_prompt": prompt}
        md_conv = MarkItDown(enable_plugins=bool(model), **kwargs)
        job.stage = "converting"
        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmp:
            path = src
            if meta["kind"] == "pdf" and len(job.pages) < meta["pages"]:
                path = docs.pdf_subset(job.file_id, job.pages, Path(tmp) / "subset.pdf")
            elif meta["kind"] == "image" and src.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                path = docs.page_image(job.file_id, job.pages[0])
            text = md_conv.convert(str(path), **kwargs).markdown
        secs = time.perf_counter() - t0 - job.load_seconds
        text = re.sub(r"^\s*# Description:\n", "", text)  # image converter's caption heading
        if o.get("drop_images", True):
            text = IMG_LINK.sub("\n\n", text)
        stats = {}
        if shim and shim.stats:
            stats = {"ocr_calls": shim.calls, "skipped_images": shim.skipped,
                     "generation_tokens": sum(r.get("generation_tokens") or 0 for r in shim.stats),
                     "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in shim.stats)}
            gen_secs = sum(r.get("seconds") or 0 for r in shim.stats)
            stats["generation_tps"] = stats["generation_tokens"] / gen_secs if gen_secs else 0
        job.whole_doc = True
        job.page_done(job.pages[0], markdown=text.strip(), text=None, seconds=secs, stats=stats)


PANDOC_READERS = {".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx", ".odt": "odt", ".rtf": "rtf",
                  ".html": "html", ".htm": "html", ".epub": "epub", ".ipynb": "ipynb", ".tex": "latex",
                  ".csv": "csv"}


class PandocEngine(Engine):
    id = "pandoc"
    name = "pandoc"
    tag = "Word with footnotes, complex tables"
    group = CONVERTERS
    description = ("pandoc 3: DOCX, PPTX, ODT, RTF, EPUB, HTML, LaTeX, ipynb, CSV → Markdown. Highest fidelity "
                   "(footnotes, nested lists, complex tables as HTML) and round-trips with Publish. "
                   "Cannot read PDFs, .xls or .msg.")
    speed = "⚡ instant"
    accepts = ("office",)

    def options(self):
        return [
            {"key": "flavor", "label": "Markdown flavor", "type": "select", "default": "gfm",
             "options": [["gfm", "GitHub (gfm)"], ["commonmark_x", "CommonMark+"], ["markdown", "pandoc markdown"]]},
            {"key": "drop_images", "label": "Drop images (saves tokens)", "type": "bool", "default": True},
        ]

    def status(self):
        return {"available": bool(shutil.which("pandoc")), "note": "" if shutil.which("pandoc") else "brew install pandoc"}

    def run(self, job):
        src = docs.source_path(job.file_id)
        reader = PANDOC_READERS.get(src.suffix.lower())
        if not reader:
            raise ValueError(f"pandoc can't read {src.suffix} files; use MarkItDown")
        cmd = ["pandoc", "-f", reader, "-t", job.opts.get("flavor") or "gfm", "--wrap=none", str(src)]
        if job.opts.get("drop_images", True):
            cmd += ["--lua-filter", str(Path(__file__).parent / "filters" / "drop_images.lua")]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode:
            hint = "\n\nTip: MarkItDown reads spreadsheets more reliably." if reader == "xlsx" else ""
            raise RuntimeError(r.stderr.strip()[-1500:] + hint)
        job.whole_doc = True
        job.page_done(job.pages[0], markdown=r.stdout.strip(), text=None, seconds=time.perf_counter() - t0)


# "What's in the file?" shortcuts in the Engines panel: each ticks the most accurate engine for one kind of
# document at under 9 s/page and names alternatives. Scores: 2026-11-06 benchmark (bench/report.html).
SUGGESTIONS = [
    {"id": "vn", "label": "Vietnamese", "pick": "mlx:mineru-2.5-pro", "why": "94.8 on Vietnamese pages, 6.5 s/page",
     "alts": [["mlx:paddleocr-vl", "faster: 91.0, 4.1 s"], ["apple-vision", "instant plain text: 91.6, no tables"],
              ["mlx:lighton-ocr-3", "most accurate: 98.8, but 15 s"]]},
    {"id": "photo", "label": "Photo of a page", "pick": "mlx:paddleocr-vl",
     "why": "89.3 on warped, skewed, badly lit and low-res captures, 4.1 s/page",
     "alts": [["mlx:unlimited-ocr-2", "88.7, 8.0 s"], ["marker", "most accurate: 93.2, but 14 s"]]},
    {"id": "en", "label": "English test / textbook", "pick": "mlx:teleocr",
     "why": "92.3 on exam papers, textbooks, slides and tables, 7.7 s/page. Drops Vietnamese",
     "alts": [["mlx:xiaomi-ocr-0", "faster: 90.4, 5.0 s"], ["mlx:paddleocr-vl", "if Vietnamese is mixed in: 89.4"]]},
    {"id": "table", "label": "Tables", "pick": "mlx:teleocr", "why": "94.7 on table structure. English and Chinese only",
     "alts": [["mlx:paddleocr-vl", "91.6, faster, reads Vietnamese"], ["mlx:xiaomi-ocr-0", "88.0, fast"]]},
    {"id": "hw", "label": "Handwriting", "pick": "apple-vision",
     "why": "instant; 86.8 on English and 77.3 on Vietnamese handwriting",
     "alts": [["mlx:paddleocr-vl", "96.0 on English lines, 65.7 on Vietnamese"],
              ["mlx:lighton-ocr-3", "79.1 on Vietnamese, 15 s"]]},
    {"id": "formula", "label": "Formulas", "pick": "mlx:paddleocr-vl",
     "why": "65.8 on LaTeX, 4.1 s/page. Every model scores 56–69 here, so check the output",
     "alts": [["mlx:glm-ocr", "67.3, 6.0 s"], ["mlx:lighton-ocr-3-1b", "68.7, 8.8 s"]]},
    {"id": "digital", "label": "Digital PDF / Office", "pick": "markitdown",
     "why": "no OCR needed: reads the file's own text instantly, in the fewest tokens",
     "alts": [["pdf-text", "PDF text with its line breaks"], ["pandoc", "Word with footnotes or complex tables"]]},
    {"id": "search", "label": "Searchable PDF", "pick": "tesseract", "why": "the only engine that exports a searchable PDF",
     "alts": []},
    {"id": "any", "label": "Not sure", "pick": "mlx:paddleocr-vl",
     "why": "the all-rounder: within a point of the best on each use case, and the fastest accurate model",
     "alts": [["mlx:unlimited-ocr-2", "long multi-page documents"]]},
]

ENGINES: dict[str, Engine] = {}
for _e in [AppleVision(), Tesseract(), *[MlxVlm(p) for p in MLX_PRESETS], Marker(), PdfText(), MarkItDownEngine(),
           PandocEngine()]:
    ENGINES[_e.id] = _e
