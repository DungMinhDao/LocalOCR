"""Uploaded document store: originals plus lazily rendered page images."""

import json
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from .paths import FILES_DIR

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
PDF_EXTS = {".pdf"}
# Digital documents for the converters (MarkItDown, pandoc); previews come from macOS Quick Look.
OFFICE_EXTS = {".docx", ".pptx", ".xlsx", ".xls", ".odt", ".rtf", ".html", ".htm", ".epub", ".csv", ".msg",
               ".ipynb", ".tex"}
_render_lock = threading.Lock()  # pdfium is not thread safe


def _meta_path(fid: str) -> Path:
    return FILES_DIR / fid / "meta.json"


def get(fid: str) -> dict:
    p = _meta_path(fid)
    if not p.exists():
        raise KeyError(fid)
    return json.loads(p.read_text())


def list_all() -> list[dict]:
    out = []
    for p in FILES_DIR.glob("*/meta.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return sorted(out, key=lambda m: m["created"], reverse=True)


def delete(fid: str) -> None:
    shutil.rmtree(FILES_DIR / fid, ignore_errors=True)


def _pdf_has_text(pdf: pdfium.PdfDocument, n: int) -> bool:
    chars = 0
    for i in range(min(n, 3)):
        tp = pdf[i].get_textpage()
        chars += len(tp.get_text_range().strip())
    return chars > 50


def add(filename: str, data: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXTS | PDF_EXTS | OFFICE_EXTS:
        raise ValueError(f"Unsupported file type {ext or '(none)'}")
    fid = uuid.uuid4().hex[:12]
    d = FILES_DIR / fid
    d.mkdir(parents=True)
    original = d / f"original{ext}"
    original.write_bytes(data)
    meta = {"id": fid, "name": filename, "ext": ext, "size": len(data), "created": time.time(),
            "kind": "pdf" if ext in PDF_EXTS else "office" if ext in OFFICE_EXTS else "image",
            "has_text_layer": ext in OFFICE_EXTS}

    if ext in {".heic", ".heif"}:  # PIL can't read HEIC; macOS sips can
        png = d / "original.png"
        subprocess.run(["sips", "-s", "format", "png", str(original), "--out", str(png)],
                       check=True, capture_output=True)
        original = png
    if meta["kind"] == "pdf":
        with _render_lock:
            pdf = pdfium.PdfDocument(str(original))
            meta["pages"] = len(pdf)
            meta["has_text_layer"] = _pdf_has_text(pdf, len(pdf))
            w, h = pdf[0].get_size()
            meta["page_size_pt"] = [round(w), round(h)]
            pdf.close()
    elif meta["kind"] == "office":
        meta["pages"] = 1
    else:
        with Image.open(original) as im:
            meta["pages"] = getattr(im, "n_frames", 1)
            meta["pixels"] = list(im.size)
    meta["source"] = original.name
    _meta_path(fid).write_text(json.dumps(meta))
    return meta


def source_path(fid: str) -> Path:
    return FILES_DIR / fid / get(fid)["source"]


def page_image(fid: str, index: int, dpi: int = 200) -> Path:
    """Return a PNG/JPEG path for page `index` (0-based). Images ignore dpi."""
    meta = get(fid)
    d = FILES_DIR / fid
    src = d / meta["source"]
    if meta["kind"] == "office":
        return _quicklook(src, d / "page-0001.png")
    if meta["kind"] == "image":
        if meta["pages"] == 1 and src.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(src) as im:
                if im.getexif().get(0x0112, 1) == 1:  # no EXIF rotation to apply
                    return src
        out = d / f"page-{index + 1:04d}.png"
        if not out.exists():
            with Image.open(src) as im:
                im.seek(index)
                ImageOps.exif_transpose(im).convert("RGB").save(out)
        return out
    out = d / f"page-{index + 1:04d}@{dpi}.png"
    if not out.exists():
        with _render_lock:
            pdf = pdfium.PdfDocument(str(src))
            img = pdf[index].render(scale=dpi / 72).to_pil()
            pdf.close()
        tmp = out.with_suffix(".tmp.png")
        img.save(tmp)
        tmp.rename(out)
    return out


def _quicklook(src: Path, out: Path) -> Path:
    """First-page preview of an office document via `qlmanage`, or a placeholder."""
    if not out.exists():
        subprocess.run(["qlmanage", "-t", "-s", "1400", "-o", str(out.parent), str(src)],
                       capture_output=True, timeout=60)
        ql = out.parent / (src.name + ".png")
        if ql.exists():
            ql.rename(out)
        else:
            from PIL import ImageDraw
            img = Image.new("RGB", (850, 1100), "white")
            ImageDraw.Draw(img).text((40, 40), f"No preview for {src.suffix} files", fill="gray")
            img.save(out)
    return out


def pdf_subset(fid: str, pages: list[int], out: Path) -> Path:
    """Write a PDF containing only `pages` (0-based)."""
    with _render_lock:
        src = pdfium.PdfDocument(str(source_path(fid)))
        dst = pdfium.PdfDocument.new()
        dst.import_pages(src, pages)
        dst.save(str(out))
        src.close()
    return out


def pdf_text(fid: str, index: int) -> str:
    with _render_lock:
        pdf = pdfium.PdfDocument(str(source_path(fid)))
        text = pdf[index].get_textpage().get_text_range()
        pdf.close()
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_pages(spec: str | None, total: int) -> list[int]:
    """'1-3,7' -> [0,1,2,6]; empty -> all pages."""
    if not spec or not spec.strip():
        return list(range(total))
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a or 1), int(b or total)
            out.extend(range(lo - 1, min(hi, total)))
        else:
            out.append(int(part) - 1)
    pages = sorted({p for p in out if 0 <= p < total})
    if not pages:
        raise ValueError(f"Page range '{spec}' selects no pages (document has {total})")
    return pages
