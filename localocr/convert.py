"""Convert OCR Markdown into the export formats."""

import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown as md_lib

PANDOC = shutil.which("pandoc")
# hard_line_breaks: OCR models emit single newlines meaning "new line", not "same paragraph".
PANDOC_FROM = "markdown+pipe_tables+raw_html+hard_line_breaks+tex_math_dollars+tex_math_single_backslash-yaml_metadata_block"

CSS = """
body{font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1d1d1f}
table{border-collapse:collapse;margin:1rem 0}td,th{border:1px solid #ccc;padding:.3rem .6rem;vertical-align:top}
th{background:#f4f4f6}pre,code{background:#f4f4f6;border-radius:4px}pre{padding:.8rem;overflow:auto}
img{max-width:100%}hr{border:0;border-top:1px solid #ddd;margin:2rem 0}
"""


def _pandoc(md: str, to: str, *extra: str, src: str = PANDOC_FROM) -> str:
    r = subprocess.run([PANDOC, "-f", src, "-t", to, "--wrap=none", *extra],
                       input=md, capture_output=True, text=True, timeout=120)
    if r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout


def to_html_fragment(md: str) -> str:
    if PANDOC:
        try:
            return _pandoc(md, "html5", "--mathml")
        except Exception:
            pass
    return md_lib.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])


def to_html(md: str, title: str) -> str:
    return (f"<!doctype html>\n<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body>\n{to_html_fragment(md)}\n</body></html>\n")


def to_text(md: str) -> str:
    if PANDOC:
        try:
            # Go through HTML so raw <table> markup from VLMs becomes real tables.
            html_doc = _pandoc(re.sub(r"<!-- page \d+ -->", "", md), "html5")
            return _pandoc(html_doc, "plain", src="html")
        except Exception:
            pass
    t = re.sub(r"</t[dh]>", "\t", md)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"(\*\*|__|`)", "", t)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    return re.sub(r"[ \t]+\n", "\n", t)


def to_docx(md: str) -> bytes:
    if not PANDOC:
        raise RuntimeError("DOCX export needs pandoc (brew install pandoc)")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.docx"
        r = subprocess.run([PANDOC, "-f", "html", "-o", str(out)], input=to_html_fragment(md),
                           capture_output=True, text=True, timeout=120)
        if r.returncode:
            raise RuntimeError(r.stderr)
        return out.read_bytes()


def searchable_pdf(images: list[Path], lang: str, dpi: int) -> bytes:
    """Image-over-text PDF produced by tesseract (selectable / searchable)."""
    with tempfile.TemporaryDirectory() as d:
        lst = Path(d) / "pages.txt"
        lst.write_text("\n".join(str(p) for p in images) + "\n")
        base = Path(d) / "out"
        r = subprocess.run(["tesseract", str(lst), str(base), "-l", lang, "--dpi", str(dpi), "pdf"],
                           capture_output=True, text=True, timeout=1800)
        if r.returncode:
            raise RuntimeError(r.stderr[-1000:])
        return base.with_suffix(".pdf").read_bytes()
