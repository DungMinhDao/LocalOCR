"""LocalOCR web server: `uv run python -m localocr.server` then open http://127.0.0.1:8765"""

import argparse
import io
import json
import re
import threading
import webbrowser
import zipfile

import uvicorn
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import convert, docs, jobs, preview, publish
from .engines import ENGINES, SUGGESTIONS, WORKERS
from .paths import STATIC

app = FastAPI(title="LocalOCR")


@app.exception_handler(KeyError)
async def _missing(_, exc):
    return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)


@app.exception_handler(ValueError)
async def _bad(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


# -- engines ---------------------------------------------------------------
@app.get("/api/engines")
def engines():
    return {"engines": [e.describe() for e in ENGINES.values()],
            "workers": {k: w.info() for k, w in WORKERS.items()},
            "pandoc": bool(convert.PANDOC), "suggestions": SUGGESTIONS}


@app.post("/api/engines/{eid}/warmup")
def warmup(eid: str):
    e = ENGINES[eid]
    threading.Thread(target=lambda: _safe(e.warmup), daemon=True).start()
    return {"ok": True}


def _safe(fn):
    try:
        fn()
    except Exception as ex:  # surfaced through worker state
        print("warmup failed:", ex)


@app.post("/api/workers/{name}/stop")
def stop_worker(name: str):
    WORKERS[name].stop()
    return WORKERS[name].info()


# -- files -----------------------------------------------------------------
@app.post("/api/files")
async def upload(files: list[UploadFile] = File(...)):
    out = []
    for f in files:
        out.append(docs.add(f.filename or "pasted.png", await f.read()))
    return out


@app.get("/api/files")
def list_files():
    return docs.list_all()


@app.delete("/api/files/{fid}")
def delete_file(fid: str):
    for j in [j for j in jobs.all_jobs() if j.file_id == fid]:
        jobs.delete(j.id)
    docs.delete(fid)
    return {"ok": True}


@app.get("/api/files/{fid}/original")
def original(fid: str):
    meta = docs.get(fid)
    return FileResponse(docs.FILES_DIR / fid / f"original{meta['ext']}", filename=meta["name"])


@app.get("/api/files/{fid}/page/{n}")
def page(fid: str, n: int, dpi: int = 120, w: int = 0):
    path = docs.page_image(fid, n, dpi)
    if w:  # thumbnails
        thumb = path.with_name(f"{path.stem}-w{w}.jpg")
        if not thumb.exists():
            with Image.open(path) as im:
                im.thumbnail((w, w * 2))
                im.convert("RGB").save(thumb, quality=80)
        path = thumb
    return FileResponse(path, headers={"Cache-Control": "max-age=86400"})


@app.get("/api/files/{fid}/text/{n}")
def text_layer(fid: str, n: int):
    """Selectable text over page n: the PDF's own text, else Apple Vision line boxes (cached)."""
    try:
        return JSONResponse(preview.text_layer(fid, n), headers={"Cache-Control": "max-age=86400"})
    except RuntimeError as ex:
        raise HTTPException(500, str(ex))


@app.get("/api/files/{fid}/view")
def html_view(fid: str):
    return preview.html_view(fid)


@app.get("/api/files/{fid}/view/{rel:path}")
def html_view_file(fid: str, rel: str):
    # Served to a sandboxed iframe; no network for Quick Look's own scripts.
    return FileResponse(preview.view_file(fid, rel), headers={"Content-Security-Policy": "connect-src 'none'"})


# -- jobs ------------------------------------------------------------------
@app.post("/api/jobs")
def create_jobs(body: dict = Body(...)):
    created, errors = [], []
    for fid in body["file_ids"]:
        for e in body["engines"]:
            try:
                created.append(jobs.submit(fid, e["id"], e.get("opts") or {}, body.get("pages")).summary())
            except ValueError as ex:
                errors.append(f"{docs.get(fid)['name']} · {ENGINES[e['id']].name}: {ex}")
    return {"jobs": created, "errors": errors}


@app.get("/api/jobs")
def list_jobs(file_id: str | None = None):
    return [j.summary() for j in jobs.all_jobs() if not file_id or j.file_id == file_id]


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return jobs.get(jid).to_dict()


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    jobs.get(jid).cancel()
    return {"ok": True}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    jobs.delete(jid)
    return {"ok": True}


def _slug(s: str) -> str:
    return re.sub(r"[^\w.-]+", "_", s).strip("_")


def _export(job, fmt: str) -> tuple[bytes | str, str, str]:
    base = f"{_slug(job.file_name.rsplit('.', 1)[0])}.{_slug(job.engine_name)}"
    md = job.markdown()
    if fmt == "md":
        return md, "text/markdown; charset=utf-8", f"{base}.md"
    if fmt == "txt":
        return job.native_text() or convert.to_text(md), "text/plain; charset=utf-8", f"{base}.txt"
    if fmt == "html":
        return convert.to_html(md, f"{job.file_name} — {job.engine_name}"), "text/html; charset=utf-8", f"{base}.html"
    if fmt == "json":
        return json.dumps(job.to_dict(), ensure_ascii=False, indent=2), "application/json", f"{base}.json"
    if fmt == "docx":
        return (convert.to_docx(md),
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document", f"{base}.docx")
    if fmt == "pdf":
        if job.engine_id != "tesseract":
            raise ValueError("Searchable PDF export is only available for Tesseract results")
        dpi = int(job.opts.get("dpi") or 300)
        imgs = [docs.page_image(job.file_id, p, dpi) for p in job.pages]
        return convert.searchable_pdf(imgs, job.opts.get("lang") or "eng", dpi), "application/pdf", f"{base}.pdf"
    raise ValueError(f"unknown format {fmt}")


@app.get("/api/jobs/{jid}/export")
def export(jid: str, fmt: str = "md", download: bool = False):
    try:
        data, mime, name = _export(jobs.get(jid), fmt)
    except RuntimeError as ex:
        raise HTTPException(500, str(ex))
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{name}"} if download else {}
    return Response(data, media_type=mime, headers=headers)


@app.get("/api/export.zip")
def export_zip(ids: str, fmt: str = "md"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for jid in ids.split(","):
            job = jobs.get(jid)
            if job.status != "done":
                continue
            data, _, name = _export(job, fmt)
            z.writestr(name, data)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f"attachment; filename=localocr-{fmt}.zip"})


@app.post("/api/render")
def render(body: dict = Body(...)):
    """Markdown → HTML using the same converter as the HTML export (for the preview)."""
    return PlainTextResponse(convert.to_html_fragment(body.get("markdown", "")))


# -- publish (Markdown → documents) ------------------------------------------
@app.get("/api/publish/formats")
def publish_formats():
    return {"formats": publish.formats(), "templates": publish.list_templates()}


@app.post("/api/publish/preview")
def publish_preview(body: dict = Body(...)):
    try:
        return PlainTextResponse(publish.preview_html(body.get("markdown", ""), body.get("options") or {}))
    except RuntimeError as ex:
        raise HTTPException(400, str(ex))


@app.post("/api/publish")
def publish_doc(body: dict = Body(...)):
    fmt = body["format"]
    try:
        data, secs = publish.convert(body.get("markdown", ""), fmt, body.get("options") or {})
    except RuntimeError as ex:
        raise HTTPException(400, str(ex))
    name = _slug(body.get("filename") or "document") + "." + publish.FORMATS[fmt]["ext"]
    return Response(data, media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{name}",
        "X-Seconds": f"{secs:.3f}", "X-Filename": name, "Access-Control-Expose-Headers": "X-Seconds, X-Filename"})


@app.post("/api/templates")
async def add_template(file: UploadFile = File(...)):
    return publish.add_template(file.filename or "template", await file.read())


@app.delete("/api/templates/{tid}")
def del_template(tid: str):
    publish.delete_template(tid)
    return {"ok": True}


@app.get("/api/templates/default/{fmt}")
def default_template(fmt: str):
    return Response(publish.default_reference(fmt), media_type="application/octet-stream",
                    headers={"Content-Disposition": f"attachment; filename=reference.{fmt}"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    jobs.load_history()
    url = f"http://{args.host}:{args.port}"
    print(f"LocalOCR running at {url}")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        for w in WORKERS.values():
            w.stop()


if __name__ == "__main__":
    main()
