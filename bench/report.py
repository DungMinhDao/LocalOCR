"""Summarise bench/results/*.jsonl into bench/report.html: accuracy per use case, speed, Pareto fronts.

    uv run --with rapidfuzz --with apted --with lxml python -m bench.report [--rescore]

The page is interactive (use-case picker, speed limit); the numbers are computed here and embedded as JSON
into bench/report_template.html. Also writes bench/results/summary.json and prints a text summary.
"""

import argparse
import json
import math
import statistics
import time

from .run import RESULTS, ROOT, samples, score

PAGE_SETS = {"omnidoc", "real5", "lowres", "vi-pages"}  # full pages: these define seconds/page

# Every sample category, in heatmap order: (id, short label, column group, metric)
CATEGORIES = [
    ("vietnamese-page", "Page", "Vietnamese", "text"),
    ("vietnamese-page-degraded", "Photo of page", "Vietnamese", "text"),
    ("vietnamese-printed-line", "Printed line", "Vietnamese", "text"),
    ("vietnamese-handwriting", "Handwriting", "Vietnamese", "text"),
    ("degraded-scanning", "Scanned", "Photos & scans", "text"),
    ("degraded-warping", "Warped", "Photos & scans", "text"),
    ("degraded-screen-photography", "Screen photo", "Photos & scans", "text"),
    ("degraded-illumination", "Bad light", "Photos & scans", "text"),
    ("degraded-skew", "Skewed", "Photos & scans", "text"),
    ("low-resolution", "Low-res", "Photos & scans", "text"),
    ("exam-paper", "Exam paper", "English documents", "text"),
    ("textbook", "Textbook", "English documents", "text"),
    ("slides", "Slides", "English documents", "text"),
    ("multi-column", "Multi-column", "English documents", "text"),
    ("table", "Tables", "English documents", "table"),
    ("formula", "Formulas", "Other", "formula"),
    ("english-handwriting", "EN handwriting", "Other", "text"),
]
METRIC = {c: m for c, _, _, m in CATEGORIES}

# Use cases: the mean of their parts (each part = one or more categories). "mine" is the owner's mix.
USE_CASES = [
    {"id": "mine", "label": "My mix", "short": "VN + photos + EN tests", "of": ["vn", "photo", "en"],
     "desc": "The average of the next three use cases, weighted equally: what this app is mostly used for."},
    {"id": "vn", "label": "Vietnamese", "short": "pages, photos, lines",
     "parts": [["vietnamese-page"], ["vietnamese-page-degraded"], ["vietnamese-printed-line"]],
     "desc": "Typeset Vietnamese pages (clean and photographed) and printed lines: diacritics must survive."},
    {"id": "photo", "label": "Photos of pages", "short": "warped, dim, low-res",
     "parts": [["degraded-scanning"], ["degraded-warping"], ["degraded-screen-photography"],
               ["degraded-illumination"], ["degraded-skew"], ["low-resolution"], ["vietnamese-page-degraded"]],
     "desc": "Pages captured by phone or scanner: warped, skewed, badly lit, photographed off a screen, low-res."},
    {"id": "en", "label": "English tests", "short": "exams, textbooks, tables",
     "parts": [["exam-paper"], ["textbook"], ["slides"], ["table"]],
     "desc": "Exam papers, textbook and lesson pages, slides, and their tables (scored on table structure)."},
    {"id": "all", "label": "All tasks", "short": "+ handwriting, formulas",
     "parts": [["multi-column", "slides", "exam-paper", "textbook"], ["table"], ["formula"],
               ["degraded-scanning", "degraded-warping", "degraded-screen-photography", "degraded-illumination",
                "degraded-skew", "low-resolution"],
               ["english-handwriting", "vietnamese-handwriting"],
               ["vietnamese-page", "vietnamese-page-degraded", "vietnamese-printed-line"]],
     "desc": "Every sample, in six groups weighted equally: printed English, tables, formulas, degraded captures, "
             "handwriting, Vietnamese. Includes handwriting and formulas, which the other use cases leave out."},
]

# Curated notes. Qualitative on purpose: the page shows the measured numbers next to them.
INFO = {
    "paddleocr-vl": {"name": "PaddleOCR-VL 1.6", "mode": "blocks", "size": "0.7 GB", "license": "Apache-2.0",
                     "best": "The default. Fastest of the accurate models and good at everything you use: "
                             "Vietnamese, photographed pages, English tests and their tables.",
                     "avoid": "Handwriting is only fair."},
    "mineru-2.5-pro": {"name": "MinerU2.5-Pro", "mode": "blocks", "size": "1.9 GB", "license": "Apache-2.0",
                       "best": "Vietnamese documents with tables; a close second to PaddleOCR-VL everywhere. "
                               "New release every month.",
                       "avoid": "Handwriting; slightly slower than PaddleOCR-VL."},
    "unlimited-ocr-2": {"name": "Unlimited-OCR-2", "mode": "whole page", "size": "2.6 GB", "license": "MIT",
                        "best": "Long or multi-page documents; strong on English tests, Vietnamese and photos.",
                        "avoid": "Handwriting: it often skips it."},
    "xiaomi-ocr-0": {"name": "Xiaomi-OCR-0", "mode": "whole page", "size": "0.65 GB", "license": "Apache-2.0",
                     "best": "English-only tests and textbooks, fast; good tables.",
                     "avoid": "Vietnamese: it drops diacritics."},
    "teleocr": {"name": "TeleOCR", "mode": "blocks", "size": "2.1 GB", "license": "Apache-2.0",
                "best": "English tests, textbooks and tables: the most accurate here.",
                "avoid": "Vietnamese and handwriting (trained on Chinese and English only)."},
    "lighton-ocr-3-1b": {"name": "LightOnOCR-3 1B", "mode": "whole page", "size": "1.4 GB", "license": "Apache-2.0",
                         "best": "Vietnamese and photographed pages; formulas.",
                         "avoid": "At the edge of the speed limit; weaker tables."},
    "lighton-ocr-3": {"name": "LightOnOCR-3 4B", "mode": "whole page", "size": "2.9 GB", "license": "Apache-2.0",
                      "best": "The most accurate Vietnamese, and good handwriting, when time doesn't matter.",
                      "avoid": "Slow; formulas."},
    "glm-ocr": {"name": "GLM-OCR", "mode": "blocks", "size": "1.6 GB", "license": "MIT",
                "best": "Formulas and tables.",
                "avoid": "Vietnamese; PaddleOCR-VL does your use cases better and faster."},
    "marker": {"name": "marker 2.1", "mode": "pipeline", "size": "~3 GB", "license": "GPL-3.0 code / OpenRAIL-M",
               "best": "Badly captured pages (the best here); digital PDFs, where it reuses the text layer.",
               "avoid": "Slow; tables."},
    "dots-mocr": {"name": "dots.mocr", "mode": "whole page", "size": "3.3 GB", "license": "MIT",
                  "best": "Clean Vietnamese and English pages.",
                  "avoid": "Photographed or degraded pages."},
    "ovisocr2": {"name": "OvisOCR2", "mode": "whole page", "size": "1.0 GB", "license": "Apache-2.0",
                 "best": "Fast whole-page Markdown with HTML tables.",
                 "avoid": "Vietnamese: Chinese characters leak in."},
    "apple-vision": {"name": "Apple Vision", "mode": "built-in", "size": "built into macOS", "license": "—",
                     "best": "Instant, no GPU: plain Vietnamese text and handwriting.",
                     "avoid": "Tables, formulas, multi-column pages and photos."},
    "tesseract": {"name": "Tesseract", "mode": "built-in", "size": "~20 MB", "license": "Apache-2.0",
                  "best": "Searchable PDF export (the only engine that makes one).",
                  "avoid": "Everything else in this comparison."},
    "chandra-ocr-2": {"name": "Chandra OCR 2", "mode": "whole page", "size": "3.6 GB", "license": "OpenRAIL-M",
                      "best": "Handwriting and Vietnamese.",
                      "avoid": "28 s/page and a large download, so it was removed from the app."},
    "glm-ocr-page": {"name": "GLM-OCR (whole page)", "mode": "whole page", "size": "", "license": "",
                     "best": "Shown for comparison: block mode is the default.",
                     "avoid": "Tables and formulas: it was trained to read one block at a time."},
    "paddleocr-vl-page": {"name": "PaddleOCR-VL (whole page)", "mode": "whole page", "size": "", "license": "",
                          "best": "Shown for comparison: block mode is the default.",
                          "avoid": "Tables and formulas: it was trained to read one block at a time."},
}


def scored_samples() -> list[dict]:
    """Samples with a ground truth to score against (one OmniDocBench notes page has none)."""
    return [s for s in samples() if s.get("gt") or s.get("gt_html") or s.get("gt_formulas")]


def load(rescore: bool) -> dict[str, list[dict]]:
    by_id = {s["id"]: s for s in scored_samples()}
    out = {}
    for f in sorted(RESULTS.glob("*.jsonl")):
        recs = []
        for line in f.read_text().split("\n"):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:  # a line still being written by a running benchmark
                pass
        recs = [r for r in recs if r["id"] in by_id]
        if rescore:
            for r in recs:
                r["scores"] = score(by_id[r["id"]], r["markdown"])
        out[f.stem] = recs
    return out


def _val(r: dict) -> float:
    v = r["scores"].get(METRIC[r["category"]])
    return 0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else v


def status(label: str) -> str:
    """featured / more / builtin / variant / removed, from the app's current engine list."""
    from localocr.engines import ENGINES

    e = ENGINES.get(label) or ENGINES.get("mlx:" + label)
    if e is None:
        return "variant" if label.endswith("-page") else "removed"
    if not label.startswith("mlx") and "mlx:" + label not in ENGINES:
        return "builtin"
    return "featured" if e.featured else "more"


def build(results: dict[str, list[dict]]) -> dict:
    counts = {c: 0 for c, *_ in CATEGORIES}
    for s in scored_samples():
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    engines = []
    for label, recs in results.items():
        cats = {}
        for c, *_ in CATEGORIES:
            vals = [_val(r) for r in recs if r["category"] == c]
            cats[c] = round(statistics.mean(vals) * 100, 1) if vals else None
        uc = {}
        for u in USE_CASES:
            if "parts" in u:
                parts = []
                for p in u["parts"]:
                    vals = [_val(r) for r in recs if r["category"] in p]
                    parts.append(statistics.mean(vals) if vals else None)
                uc[u["id"]] = round(statistics.mean(parts) * 100, 1) if None not in parts else None
        mine = [uc[k] for k in USE_CASES[0]["of"]]
        uc["mine"] = round(statistics.mean(mine), 1) if None not in mine else None
        secs = [r["seconds"] for r in recs if r["set"] in PAGE_SETS and r["status"] == "done"]
        info = INFO.get(label, {"name": label, "best": "", "avoid": ""})
        engines.append({"id": label, "status": status(label), "sec": round(statistics.median(secs), 2) if secs else None,
                        "cats": cats, "uc": uc, "samples": len(recs), **info})
    return {
        "generated": time.strftime("%Y-%m-%d"), "n": len(scored_samples()),
        "categories": [{"id": c, "label": l, "group": g, "metric": m, "n": counts.get(c, 0)}
                       for c, l, g, m in CATEGORIES],
        "useCases": [{k: u[k] for k in ("id", "label", "short", "desc")} for u in USE_CASES],
        "engines": engines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="recompute scores (after changing metrics)")
    args = ap.parse_args()
    data = build(load(args.rescore))
    (RESULTS / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    page = (ROOT / "report_template.html").read_text()
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    (ROOT / "report.html").write_text(page.replace("/*DATA*/null", blob))

    ucs = [u["id"] for u in USE_CASES]
    print(f"{'engine':20} {'s/page':>6}  " + "  ".join(f"{u:>6}" for u in ucs) + "  status")
    for e in sorted(data["engines"], key=lambda e: -(e["uc"]["mine"] or 0)):
        sc = "  ".join(f"{'–' if e['uc'][u] is None else e['uc'][u]:>6}" for u in ucs)
        print(f"{e['id']:20} {e['sec'] or 0:6.1f}  {sc}  {e['status']}")
    print(f"wrote {ROOT / 'report.html'}")


if __name__ == "__main__":
    main()
