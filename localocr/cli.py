"""Command line for agents and scripts: `./localocr extract|publish|engines` (see `--help`).

extract reuses the running server (warm models) when it answers on LOCALOCR_URL, otherwise it runs the
engine in-process and stops any worker it started. publish always runs pandoc in-process.
"""

import argparse
import json
import mimetypes
import os
import signal
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SERVER = os.environ.get("LOCALOCR_URL", "http://127.0.0.1:8765").rstrip("/")
FORMATS = ("md", "txt", "json")


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def die(msg: str, code: int = 1):
    log(f"localocr: {msg}")
    sys.exit(code)


def parse_opts(pairs: list[str]) -> dict:
    """--opt dpi=300 --opt langs=vi-VT,en-US: values are JSON when they parse (numbers, true/false)."""
    out = {}
    for p in pairs or []:
        k, sep, v = p.partition("=")
        if not sep:
            die(f"--opt expects key=value, got {p!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def resolve_engine(name: str, ids: list[str]) -> str:
    if name in ids:
        return name
    if f"mlx:{name}" in ids:  # `glm-ocr` for `mlx:glm-ocr`
        return f"mlx:{name}"
    die(f"unknown engine {name!r}; run `localocr engines` for the list")


def auto_engine(meta: dict) -> str:
    """Cheapest engine that does the job: converters for digital files, Apple Vision for scans."""
    if meta["kind"] == "office":
        return "markitdown"
    if meta["kind"] == "pdf" and meta.get("has_text_layer"):
        return "pdf-text"
    return "apple-vision"


# -- server backend ----------------------------------------------------------
class Remote:
    def __init__(self, url: str):
        self.url = url

    def _req(self, method: str, path: str, data: bytes | None = None, ctype: str | None = None, timeout=30):
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except Exception:
                detail = e.reason
            raise RuntimeError(f"server: {detail}") from None

    def json(self, method: str, path: str, body=None):
        data = json.dumps(body).encode() if body is not None else None
        return json.loads(self._req(method, path, data, "application/json" if data else None))

    def alive(self) -> bool:
        try:
            self._req("GET", "/api/engines", timeout=0.5)
            return True
        except Exception:
            return False

    def engines(self) -> list[dict]:
        return self.json("GET", "/api/engines")["engines"]

    def upload(self, path: Path) -> dict:
        boundary = uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = (f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{path.name}"\r\n'
                f"Content-Type: {ctype}\r\n\r\n").encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        out = self._req("POST", "/api/files", body, f"multipart/form-data; boundary={boundary}", timeout=300)
        return json.loads(out)[0]

    def submit(self, fid: str, engine: str, opts: dict, pages: str | None) -> str:
        r = self.json("POST", "/api/jobs", {"file_ids": [fid], "engines": [{"id": engine, "opts": opts}],
                                            "pages": pages})
        if r["errors"]:
            raise RuntimeError(r["errors"][0])
        return r["jobs"][0]["id"]

    def status(self, jid: str) -> dict:
        return self.json("GET", f"/api/jobs/{jid}")

    def output(self, jid: str, fmt: str) -> str:
        return self._req("GET", f"/api/jobs/{jid}/export?fmt={fmt}").decode()

    def cancel(self, jid: str) -> None:
        self.json("POST", f"/api/jobs/{jid}/cancel")

    def forget(self, fid: str) -> None:
        self.json("DELETE", f"/api/files/{fid}")

    def close(self) -> None:
        pass


# -- in-process backend --------------------------------------------------------
class Local:
    def __init__(self):
        from . import docs, jobs
        from .engines import ENGINES, WORKERS
        self.docs, self.jobs, self.ENGINES, self.WORKERS = docs, jobs, ENGINES, WORKERS

    def engines(self) -> list[dict]:
        return [e.describe() for e in self.ENGINES.values()]

    def upload(self, path: Path) -> dict:
        return self.docs.add(path.name, path.read_bytes())

    def submit(self, fid, engine, opts, pages) -> str:
        return self.jobs.submit(fid, engine, opts, pages).id

    def status(self, jid: str) -> dict:
        return self.jobs.get(jid).summary()

    def output(self, jid: str, fmt: str) -> str:
        from . import convert
        job = self.jobs.get(jid)
        if fmt == "txt":
            return job.native_text() or convert.to_text(job.markdown())
        if fmt == "json":
            return json.dumps(job.to_dict(), ensure_ascii=False, indent=2)
        return job.markdown()

    def cancel(self, jid: str) -> None:
        self.jobs.get(jid).cancel()

    def forget(self, fid: str) -> None:
        for j in [j for j in self.jobs.all_jobs() if j.file_id == fid]:
            self.jobs.delete(j.id)
        self.docs.delete(fid)

    def close(self) -> None:
        for w in self.WORKERS.values():
            w.stop()


def backend(args):
    if not args.local:
        r = Remote(args.server)
        if r.alive():
            return r
        if args.server != SERVER:  # an explicit --server that isn't up is an error, not a fallback
            die(f"no LocalOCR server at {args.server}")
    return Local()


# -- commands ------------------------------------------------------------------
def cmd_engines(args) -> None:
    be = backend(args)
    try:
        es = be.engines()
    finally:
        be.close()
    if args.json:
        print(json.dumps(es, ensure_ascii=False, indent=2))
        return
    for e in es:
        eid = e["id"].removeprefix("mlx:")
        flag = " " if e.get("available", True) else "✗"
        note = f"  ({e['note']})" if e.get("note") else ""
        print(f"{flag} {eid:16} {e['name']:22} {','.join(e['accepts']):18} {e.get('speed', '')}{note}")
        if args.verbose:
            for o in e["options"]:
                print(f"      --opt {o['key']}=…  {o['label']} (default {o.get('default')!r})")


def _wait(be, jid: str, label: str, quiet: bool) -> dict:
    shown = ""
    while True:
        s = be.status(jid)
        if s["status"] not in ("queued", "running"):
            if shown:
                log()
            return s
        if not quiet:
            n = len(s["pages"])
            line = f"{label}: {s['status']}" + (f" · page {s['pages_done'] + 1}/{n}" if n > 1 and
                                                s["status"] == "running" else "")
            if line != shown:
                print(f"\r\033[K{line}", end="", file=sys.stderr, flush=True)
                shown = line
        time.sleep(0.15)


def cmd_extract(args) -> None:
    if args.format not in FORMATS:
        die(f"--format must be one of {', '.join(FORMATS)}")
    files = [Path(f) for f in args.files]
    for f in files:
        if not f.is_file():
            die(f"no such file: {f}")
    outdir = Path(args.output) if args.output and (len(files) > 1 or Path(args.output).is_dir()) else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
    opts = parse_opts(args.opt)
    quiet = args.quiet or not sys.stderr.isatty()

    be = backend(args)
    if not quiet:
        log(f"localocr: using {'server ' + be.url if isinstance(be, Remote) else 'in-process engines'}")
    ids = [e["id"] for e in be.engines()] if args.engine != "auto" else []
    current: dict = {}

    def interrupt(*_):
        if current.get("jid"):
            try:
                be.cancel(current["jid"])
            except Exception:
                pass
        cleanup()
        be.close()
        os._exit(130)

    def cleanup():
        if current.get("fid") and not args.keep:
            try:
                be.forget(current["fid"])
            except Exception:
                pass

    signal.signal(signal.SIGINT, interrupt)
    failed = 0
    try:
        for i, f in enumerate(files):
            meta = be.upload(f)
            current.update(fid=meta["id"], jid=None)
            try:
                eid = auto_engine(meta) if args.engine == "auto" else resolve_engine(args.engine, ids)
                pages = None if meta["kind"] == "office" else args.pages  # Office files have no pages
                current["jid"] = jid = be.submit(meta["id"], eid, opts, pages)
                s = _wait(be, jid, f"{f.name} · {eid.removeprefix('mlx:')}", quiet)
                if s["status"] != "done":
                    raise RuntimeError(s["error"] or s["status"])
                text = be.output(jid, args.format)
                if not quiet:
                    log(f"{f.name}: {s['engine_name']} · {s['total_seconds']:.1f} s · ≈{s['tokens_est']} tokens")
            except Exception as ex:
                msg = str(ex).removeprefix("server: ")
                log(f"localocr: {msg if msg.startswith(f.name) else f'{f.name}: {msg}'}")
                failed += 1
                continue
            finally:
                cleanup()
            if outdir:
                out = outdir / f"{f.stem}.{args.format}"
                out.write_text(text)
                log(str(out))
            elif args.output:
                Path(args.output).write_text(text)
            else:
                if len(files) > 1 and args.format != "json":
                    sys.stdout.write(f"{'' if i == 0 else chr(10)}<!-- file: {f.name} -->\n\n")
                sys.stdout.write(text if text.endswith("\n") else text + "\n")
                sys.stdout.flush()
    finally:
        be.close()
    sys.exit(1 if failed else 0)


def cmd_publish(args) -> None:
    from . import publish

    if args.to not in publish.FORMATS:
        die(f"--to must be one of {', '.join(publish.FORMATS)}")
    if not publish.formats()[args.to]["available"]:
        die(f"{args.to} is unavailable: {publish.formats()[args.to]['note']}")
    md = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
    opts = {"toc": args.toc, "number_sections": args.number_sections, "title": args.title,
            "hard_breaks": args.hard_breaks, "slide_level": args.slide_level}
    if args.template:
        opts["reference_doc"] = str(_template(args.template, args.to, publish))
    try:
        data, secs = publish.convert(md, args.to, opts)
    except (RuntimeError, ValueError) as ex:
        die(str(ex))
    out = args.output or ("-" if args.input == "-" else
                          str(Path(args.input).with_suffix("." + publish.FORMATS[args.to]["ext"])))
    if out == "-":
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    else:
        Path(out).write_bytes(data)
        log(f"{out} ({len(data) / 1024:.0f} KB, {secs:.1f} s)")


def _template(spec: str, fmt: str, publish) -> Path:
    """A reference-doc path, or the name/id of a template uploaded in the GUI."""
    p = Path(spec).expanduser()
    if p.is_file():
        return p.resolve()
    for t in publish.list_templates():
        if t["format"] == fmt and spec in (t["id"], t["name"], Path(t["name"]).stem):
            return publish.TEMPLATES_DIR / f"{t['id']}.{fmt}"
    die(f"no {fmt} template {spec!r} (give a file path, or the name of a template uploaded in the GUI)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="localocr", description="Local documents ⇄ Markdown, for LLMs and agents.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def backend_flags(p):
        p.add_argument("--server", default=SERVER, help=f"LocalOCR server to reuse (default {SERVER}, $LOCALOCR_URL)")
        p.add_argument("--local", action="store_true", help="run in-process even if the server is up")

    p = sub.add_parser("extract", help="scan / PDF / Office file → Markdown on stdout",
                       description="Extract Markdown from files. Progress goes to stderr, output to stdout.")
    p.add_argument("files", nargs="+")
    p.add_argument("-e", "--engine", default="auto",
                   help="engine id from `localocr engines` (e.g. apple-vision, tesseract, glm-ocr, dots-mocr, "
                        "marker, pdf-text, markitdown, pandoc). auto: markitdown for Office files, pdf-text for "
                        "digital PDFs, apple-vision for scans and images")
    p.add_argument("-p", "--pages", help="1-based page range, e.g. 1-3,7 (default: all)")
    p.add_argument("--opt", action="append", metavar="KEY=VALUE",
                   help="engine option, repeatable; see `localocr engines -v`")
    p.add_argument("-f", "--format", default="md", help="md (default), txt, or json (with timings)")
    p.add_argument("-o", "--output", help="output file; a directory when given several input files")
    p.add_argument("--keep", action="store_true", help="keep the file and result in the GUI's history")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress on stderr")
    backend_flags(p)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("publish", help="Markdown → docx, pptx, pdf, xlsx, html, odt, epub, latex, typst, gfm",
                       description="Convert Markdown with pandoc. Writes next to the input unless -o is given.")
    p.add_argument("input", help="Markdown file, or - for stdin")
    p.add_argument("-t", "--to", required=True, help="docx, pptx, pdf, xlsx, html, odt, epub, latex, typst, gfm")
    p.add_argument("-o", "--output", help="output file, or - for stdout (default: input name with the new extension)")
    p.add_argument("--template", help="reference .docx/.pptx/.odt path, or the name of a template uploaded in the GUI")
    p.add_argument("--title")
    p.add_argument("--toc", action="store_true", help="table of contents")
    p.add_argument("--number-sections", action="store_true")
    p.add_argument("--hard-breaks", action="store_true", help="keep single line breaks")
    p.add_argument("--slide-level", type=int, help="pptx: heading level that starts a new slide")
    p.set_defaults(fn=cmd_publish)

    p = sub.add_parser("engines", help="list engines, their ids and availability")
    p.add_argument("-v", "--verbose", action="store_true", help="also list each engine's --opt keys")
    p.add_argument("--json", action="store_true")
    backend_flags(p)
    p.set_defaults(fn=cmd_engines)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
