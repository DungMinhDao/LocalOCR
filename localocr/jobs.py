"""OCR jobs: one (file, engine, options, pages) run, executed in background threads."""

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from . import docs
from .engines import ENGINES, MLX_PRESETS
from .paths import JOBS_DIR

# GPU engines share one lane so their timings aren't distorted by each other.
_gpu = ThreadPoolExecutor(1, thread_name_prefix="gpu")
_cpu = ThreadPoolExecutor(4, thread_name_prefix="cpu")
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()


class Job:
    def __init__(self, file_id: str, engine_id: str, opts: dict, pages: list[int], **kw):
        meta = docs.get(file_id)
        engine = ENGINES[engine_id]
        self.id = kw.get("id") or uuid.uuid4().hex[:12]
        self.file_id = file_id
        self.file_name = meta["name"]
        self.file_size = meta["size"]
        self.engine_id = engine_id
        self.engine_name = engine.name
        self.opts = {**engine.default_opts(), **(opts or {})}
        self.pages = pages
        self.status = kw.get("status", "queued")
        self.stage = ""
        self.created = kw.get("created", time.time())
        self.started = kw.get("started")
        self.finished = kw.get("finished")
        self.load_seconds = kw.get("load_seconds", 0.0)
        self.results: dict[int, dict] = {int(k): v for k, v in kw.get("results", {}).items()}
        self.error = kw.get("error", "")
        self.partial = ""
        self.current_page = None
        self.cancelled = False
        self.cancel_hook = None
        self.proc = None
        self.whole_doc = kw.get("whole_doc", False)  # one result covering all selected pages
        self.model = kw.get("model") or (engine.model_id(self.opts) if hasattr(engine, "model_id") else None)

    # called by engines
    def page_done(self, page: int, markdown: str, text: str | None, seconds: float, stats: dict | None = None):
        self.results[page] = {"markdown": markdown, "text": text, "seconds": seconds, "stats": stats or {},
                              "chars": len(markdown or text or "")}
        self.partial = ""

    # assembled outputs
    def markdown(self) -> str:
        parts = []
        multi = len(self.pages) > 1 and not self.whole_doc
        for p in sorted(self.results):
            if multi:
                parts.append(f"<!-- page {p + 1} -->")
            parts.append(self.results[p]["markdown"] or "")
        return "\n\n".join(parts).strip() + "\n"

    def native_text(self) -> str | None:
        rs = [self.results[p] for p in sorted(self.results)]
        if rs and all(r.get("text") is not None for r in rs):
            return "\n\n\f".join(r["text"] for r in rs).strip() + "\n"
        return None

    @property
    def total_seconds(self):
        if self.started:
            return (self.finished or time.time()) - self.started
        return None

    def summary(self) -> dict:
        ocr_secs = sum(r["seconds"] for r in self.results.values())
        n = len(self.pages) if self.whole_doc and self.results else len(self.results)
        toks = sum((r["stats"] or {}).get("generation_tokens") or 0 for r in self.results.values())
        gen_secs = sum(r["seconds"] for r in self.results.values() if (r["stats"] or {}).get("generation_tokens"))
        return {
            "id": self.id, "file_id": self.file_id, "file_name": self.file_name, "engine_id": self.engine_id,
            "engine_name": self.engine_name, "model": self.model, "opts": self.opts, "pages": self.pages,
            "status": self.status, "stage": self.stage, "error": self.error, "created": self.created,
            "started": self.started, "finished": self.finished, "total_seconds": self.total_seconds,
            "load_seconds": self.load_seconds, "ocr_seconds": ocr_secs, "pages_done": n,
            "seconds_per_page": (ocr_secs / n) if n else None,
            "tokens": toks or None, "tokens_per_second": (toks / gen_secs) if toks and gen_secs else None,
            "chars": sum(r["chars"] for r in self.results.values()),
            # ≈ LLM tokens: ~4 bytes of UTF-8 per token (accented / CJK text costs more, as it should)
            "tokens_est": round(sum(len((r["markdown"] or "").encode()) for r in self.results.values()) / 4),
            "file_size": self.file_size,
            "current_page": self.current_page, "whole_doc": self.whole_doc,
        }

    def to_dict(self) -> dict:
        d = self.summary()
        d["results"] = {str(k): v for k, v in sorted(self.results.items())}
        d["partial"] = self.partial
        return d

    def save(self) -> None:
        d = self.to_dict()
        d.pop("partial", None)
        (JOBS_DIR / f"{self.id}.json").write_text(json.dumps(d, ensure_ascii=False))

    def cancel(self) -> None:
        self.cancelled = True
        if self.status == "queued":
            self.status = "cancelled"
        if self.cancel_hook:
            self.cancel_hook()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def _run(job: Job) -> None:
    if job.cancelled:
        job.status = "cancelled"
        job.save()
        return
    job.status, job.started, job.stage = "running", time.time(), "running"
    try:
        ENGINES[job.engine_id].run(job)
        job.status = "cancelled" if job.cancelled else "done"
    except Exception as e:
        traceback.print_exc()
        job.status, job.error = ("cancelled", "") if job.cancelled else ("error", f"{e}")
    finally:
        job.finished, job.stage, job.partial, job.current_page = time.time(), "", "", None
        job.cancel_hook = None
        job.save()


def submit(file_id: str, engine_id: str, opts: dict, page_spec: str | None) -> Job:
    meta = docs.get(file_id)
    engine = ENGINES[engine_id]
    if meta["kind"] not in engine.accepts:
        raise ValueError(f"{engine.name} only accepts {', '.join(engine.accepts)} files")
    job = Job(file_id, engine_id, opts, docs.parse_pages(page_spec, meta["pages"]))
    if engine_id == "markitdown":
        job.model = job.opts.get("ocr_model") or None
        preset = next((p for p in MLX_PRESETS if p["model"] == job.model), None)
        job.engine_name = f"MarkItDown + {preset['name'] if preset else job.model.split('/')[-1]}" if job.model else "MarkItDown"
    with _lock:
        _jobs[job.id] = job
    (_gpu if engine.uses_gpu(job.opts) else _cpu).submit(_run, job)
    return job


def get(jid: str) -> Job:
    return _jobs[jid]


def all_jobs() -> list[Job]:
    return sorted(_jobs.values(), key=lambda j: j.created, reverse=True)


def delete(jid: str) -> None:
    job = _jobs.pop(jid, None)
    if job:
        job.cancel()
    (JOBS_DIR / f"{jid}.json").unlink(missing_ok=True)


def load_history() -> None:
    for p in JOBS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            if d["engine_id"] not in ENGINES:
                continue
            docs.get(d["file_id"])
            if d["status"] in ("queued", "running"):
                d["status"], d["error"] = "error", "interrupted (server restarted)"
            fields = {k: d.get(k) for k in ("id", "status", "created", "started", "finished", "load_seconds",
                                            "results", "error", "model", "whole_doc")}
            job = _jobs[d["id"]] = Job(d["file_id"], d["engine_id"], d["opts"], d["pages"],
                                 **{k: v for k, v in fields.items() if v is not None})
            job.engine_name = d.get("engine_name") or job.engine_name
        except Exception:
            continue
