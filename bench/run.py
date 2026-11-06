"""Run OCR engines over the benchmark samples through the running LocalOCR server (warm models).

    ./start.sh --no-browser &
    uv run --with rapidfuzz --with apted --with lxml python -m bench.run apple-vision glm-ocr [--limit N] [--set S]

Each sample is one page image with ground truth (bench/data/*/manifest.jsonl). Results are appended to
bench/results/<engine>.jsonl and skipped on re-runs, so an interrupted run resumes. Timing is the engine's
own per-page time (model load excluded; the first page per engine is a discarded warm-up).
"""

import argparse
import json
import time
from pathlib import Path

from localocr.cli import SERVER, Remote, parse_opts, resolve_engine

from .metrics import formula_page_score, teds, text_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
TESS_LANG = {"en": "eng", "vi": "vie"}


def samples(sets: list[str] | None = None) -> list[dict]:
    out = []
    for m in sorted(DATA.glob("*/manifest.jsonl")):
        if sets and m.parent.name not in sets:
            continue
        for line in m.read_text().split("\n"):
            if line.strip():
                s = json.loads(line)
                s["image"] = str(m.parent / s["image"])
                out.append(s)
    return out


def score(s: dict, md: str) -> dict:
    r = {"text": text_score(md, s["gt"])}
    if s["kind"] == "table":
        r["table"] = teds(md, s.get("gt_html") or s["gt"])
    if s["kind"] == "formula":
        r["formula"] = formula_page_score(md, s["gt_formulas"])
    return r


def run_engine(remote: Remote, label: str, eid: str, opts: dict, todo: list[dict], uploads: dict) -> None:
    out = RESULTS / f"{label}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().split("\n") if l.strip()} if out.exists() else set()
    todo = [s for s in todo if s["id"] not in done]
    if not todo:
        print(f"{label}: nothing to do")
        return
    warm = True
    for k, s in enumerate(todo):
        if s["image"] not in uploads:
            uploads[s["image"]] = remote.upload(Path(s["image"]))["id"]
        fid = uploads[s["image"]]
        # "auto" language options follow the sample's language (e.g. Tesseract eng / vie)
        o = {k: (TESS_LANG.get(s.get("lang"), "eng") if v == "auto" else v) for k, v in opts.items()}
        for attempt in ("warm-up", "run") if warm else ("run",):
            jid = remote.submit(fid, eid, o, None)
            while (st := remote.status(jid))["status"] in ("queued", "running"):
                time.sleep(0.1)
        warm = False
        md = remote.output(jid, "md") if st["status"] == "done" else ""
        rec = {"id": s["id"], "set": s["set"], "category": s["category"], "kind": s["kind"],
               "lang": s.get("lang"), "engine": label, "status": st["status"], "error": st.get("error"),
               "seconds": st["ocr_seconds"], "tokens_out": st.get("tokens"), "markdown": md,
               "scores": score(s, md)}
        with out.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        sc = " ".join(f"{k}={v:.2f}" for k, v in rec["scores"].items())
        print(f"{label} [{k + 1}/{len(todo)}] {s['id']}: {rec['seconds']:.2f}s {sc} {st['status']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engines", nargs="+", help="engine id, optionally label=engine and engine@key=val,key=val")
    ap.add_argument("--set", action="append", help="only these dataset folders")
    ap.add_argument("--limit", type=int, help="first N samples per category")
    ap.add_argument("--server", default=SERVER)
    args = ap.parse_args()

    remote = Remote(args.server)
    if not remote.alive():
        raise SystemExit(f"start the server first ({args.server})")
    ids = [e["id"] for e in remote.engines()]
    todo = samples(args.set)
    if args.limit:
        per: dict[str, int] = {}
        todo = [s for s in todo if per.setdefault(s["category"], 0) < args.limit
                and not per.update({s["category"]: per[s["category"]] + 1})]
    RESULTS.mkdir(exist_ok=True)
    uploads: dict[str, str] = {}
    try:
        for spec in args.engines:
            head, _, optstr = spec.partition("@")
            label, _, name = head.rpartition("=")
            opts = parse_opts(optstr.split(",") if optstr else [])
            run_engine(remote, label or name.replace(":", "-"), resolve_engine(name, ids), opts, todo, uploads)
    finally:
        for fid in uploads.values():
            remote.forget(fid)


if __name__ == "__main__":
    main()
