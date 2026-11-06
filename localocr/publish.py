"""Markdown → documents (pandoc GUI backend), plus reference-doc templates."""

import io
import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .convert import CSS, PANDOC
from .paths import DATA

TEMPLATES_DIR = DATA / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

# Authored Markdown, not OCR output: normal paragraph rules, YAML front matter allowed.
FROM = "markdown+pipe_tables+raw_html+tex_math_dollars+emoji"
CHROME = next((p for p in ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                           "/Applications/Chromium.app/Contents/MacOS/Chromium",
                           "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"] if Path(p).exists()), None)

FORMATS = {
    "docx": {"label": "Word", "ext": "docx", "template": True},
    "pptx": {"label": "PowerPoint", "ext": "pptx", "template": True, "slides": True},
    "pdf": {"label": "PDF", "ext": "pdf"},
    "xlsx": {"label": "Excel (tables)", "ext": "xlsx"},
    "html": {"label": "HTML (standalone)", "ext": "html"},
    "odt": {"label": "OpenDocument", "ext": "odt", "template": True},
    "epub": {"label": "EPUB", "ext": "epub"},
    "latex": {"label": "LaTeX", "ext": "tex"},
    "typst": {"label": "Typst", "ext": "typ"},
    "gfm": {"label": "Markdown (normalised GFM)", "ext": "md"},
}


def pdf_engine() -> str | None:
    if shutil.which("typst"):
        return "typst"
    if CHROME:
        return "chrome"
    return None


def formats() -> dict:
    out = {}
    for k, v in FORMATS.items():
        ok, note = bool(PANDOC), "" if PANDOC else "brew install pandoc"
        if k == "pdf":
            eng = pdf_engine()
            ok = ok and eng is not None
            note = {"typst": "via Typst", "chrome": "via headless Chrome · `brew install typst` for nicer typesetting",
                    None: "Needs Typst (brew install typst) or Chrome"}[eng]
        out[k] = {**v, "available": ok, "note": note}
    return out


# -- templates ---------------------------------------------------------------
def list_templates() -> list[dict]:
    out = []
    for p in sorted(TEMPLATES_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def add_template(name: str, data: bytes) -> dict:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext not in ("docx", "pptx", "odt"):
        raise ValueError("Reference templates must be .docx, .pptx or .odt")
    tid = uuid.uuid4().hex[:10]
    (TEMPLATES_DIR / f"{tid}.{ext}").write_bytes(data)
    meta = {"id": tid, "name": name, "format": ext}
    (TEMPLATES_DIR / f"{tid}.json").write_text(json.dumps(meta))
    return meta


def delete_template(tid: str) -> None:
    for p in TEMPLATES_DIR.glob(f"{tid}.*"):
        p.unlink()


def default_reference(fmt: str) -> bytes:
    """pandoc's built-in reference doc, to be restyled in Word/PowerPoint and uploaded back."""
    if fmt not in ("docx", "pptx", "odt"):
        raise ValueError(fmt)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / f"reference.{fmt}"
        subprocess.run([PANDOC, "-o", str(out), "--print-default-data-file", f"reference.{fmt}"],
                       capture_output=True, check=True)
        return out.read_bytes()


# -- conversion ----------------------------------------------------------------
def _args(opts: dict) -> list[str]:
    a = []
    if opts.get("toc"):
        a.append("--toc")
    if opts.get("number_sections"):
        a.append("--number-sections")
    if opts.get("title"):
        a += ["--metadata", f"title={opts['title']}"]
    return a


def _from(opts: dict) -> str:
    return FROM + ("+hard_line_breaks" if opts.get("hard_breaks") else "")


def preview_html(md: str, opts: dict) -> str:
    r = subprocess.run([PANDOC, "-f", _from(opts), "-t", "html5", "--mathml", "--section-divs", *_args(opts)],
                       input=md, capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


def standalone_html(md: str, opts: dict) -> str:
    title = opts.get("title") or "Document"
    r = subprocess.run([PANDOC, "-f", _from(opts), "-t", "html5", "--standalone", "--mathml", *_args(opts),
                        "--metadata", f"pagetitle={title}", "--variable", "maxwidth=52em"],
                       input=md, capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.replace("</head>", f"<style>{CSS}@media print{{body{{max-width:none;margin:0}}}}</style></head>", 1)


def _tables_xlsx(md: str, opts: dict) -> bytes:
    import pandas as pd

    html = preview_html(md, {**opts, "toc": False})
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        raise ValueError("No tables found in the Markdown to put into Excel")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for i, df in enumerate(tables, 1):
            df.to_excel(xw, sheet_name=f"Table {i}", index=False)
            ws = xw.sheets[f"Table {i}"]
            for col in ws.columns:  # rough auto-width
                width = max(len(str(c.value or "")) for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(60, max(8, width + 2))
    return buf.getvalue()


def convert(md: str, fmt: str, opts: dict) -> tuple[bytes, float]:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt}")
    t0 = time.perf_counter()
    if fmt == "xlsx":
        return _tables_xlsx(md, opts), time.perf_counter() - t0
    if fmt == "html":
        return standalone_html(md, opts).encode(), time.perf_counter() - t0

    with tempfile.TemporaryDirectory() as d:
        if fmt == "pdf" and pdf_engine() == "chrome":
            src = Path(d) / "doc.html"
            src.write_text(standalone_html(md, opts))
            out = Path(d) / "doc.pdf"
            r = subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={d}/prof",
                                "--no-pdf-header-footer", f"--print-to-pdf={out}", src.as_uri()],
                               capture_output=True, text=True, timeout=120)
            if not out.exists():
                raise RuntimeError(r.stderr[-1500:] or "Chrome produced no PDF")
            return out.read_bytes(), time.perf_counter() - t0

        out = Path(d) / f"doc.{FORMATS[fmt]['ext']}"
        cmd = [PANDOC, "-f", _from(opts), "-t", fmt, "--standalone", "-o", str(out), *_args(opts)]
        if fmt == "pdf":
            cmd += ["--pdf-engine=typst"]
        if fmt == "gfm":
            cmd += ["--wrap=none"]
        if fmt == "pptx" and opts.get("slide_level"):
            cmd += [f"--slide-level={int(opts['slide_level'])}"]
        tid = opts.get("template")
        if FORMATS[fmt].get("template") and (tid or opts.get("reference_doc")):
            ref = Path(opts["reference_doc"]) if opts.get("reference_doc") else TEMPLATES_DIR / f"{tid}.{fmt}"
            if ref.exists():
                cmd += [f"--reference-doc={ref}"]
        r = subprocess.run(cmd, input=md, capture_output=True, text=True, timeout=300, cwd=d)
        if r.returncode:
            raise RuntimeError(r.stderr.strip()[-1500:])
        return out.read_bytes(), time.perf_counter() - t0
