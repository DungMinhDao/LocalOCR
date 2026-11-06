"""Download a small, stratified OCR test set with ground truth into bench/data/<set>/manifest.jsonl.

    uv run --with pandas --with pyarrow --with pillow --with pypdfium2 python -m bench.fetch

Sets (English unless noted; ~110 samples):
  omnidoc   OmniDocBench v1.6 pages: tables, formulas, multi-column, slides, notes, exam papers, textbooks
  real5     the same pages photographed/scanned/warped/skewed/badly lit (Real5-OmniDocBench)
  lowres    OmniDocBench pages downscaled to ~40 % (≈ 70–90 dpi)
  vi-pages  Vietnamese Wikipedia text typeset by LocalOCR's publish pipeline: clean and degraded scans
  vi-lines  Vietnamese handwriting (HenryBui/…line_ocr, manhha2502/…Handwriting_OCR) and printed lines\n            (ihbkaiser/synthetic-vi-news-ocr)
  iam       English handwriting lines (Teklia/IAM-line)
"""

import io
import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).parent
RAW = ROOT / "raw"
DATA = ROOT / "data"
HF = "https://huggingface.co/datasets"
rng = random.Random(7)

# OmniDocBench evaluation ignores these regions; so do we.
SKIP = {"header", "footer", "page_number", "page_footnote", "abandon", "figure", "table_mask", "text_mask",
        "need_mask", "chart"}


def get(url: str, dest: Path | None = None) -> bytes:
    if dest and dest.exists():
        return dest.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "LocalOCR-bench"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    if dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def write_set(name: str, rows: list[dict]) -> None:
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    with (d / "manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"set": name, **r}, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} samples")


# -- OmniDocBench ------------------------------------------------------------------
def omnidoc_gt(page: dict) -> dict:
    dets = sorted((d for d in page["layout_dets"] if not d.get("ignore") and d["category_type"] not in SKIP
                   and d.get("order") is not None), key=lambda d: d["order"])
    md, tables, formulas = [], [], []
    for d in dets:
        c = d["category_type"]
        if c == "table":
            html = d.get("html") or ""
            tables.append(html)
            md.append(html)
        elif c == "equation_isolated":
            latex = d.get("latex") or d.get("text") or ""
            formulas.append(latex)
            md.append(f"$$\n{latex.strip('$')}\n$$")
        elif c == "title":
            md.append("# " + (d.get("text") or ""))
        elif d.get("text"):
            md.append(d["text"])
    return {"gt": "\n\n".join(md), "tables": tables, "formulas": formulas}


def omnidoc() -> list[dict]:
    pages = json.loads(get(f"{HF}/opendatalab/OmniDocBench/resolve/main/OmniDocBench.json", RAW / "omnidoc.json"))
    en = [p for p in pages if p["page_info"]["page_attribute"]["language"] == "english"]
    for p in en:
        p["_gt"] = omnidoc_gt(p)
        p["_attr"] = p["page_info"]["page_attribute"]

    def pick(cond, n):
        pool = [p for p in en if cond(p) and not p.get("_used")]
        out = rng.sample(pool, min(n, len(pool)))
        for p in out:
            p["_used"] = True
        return out

    plan = [
        ("table", "table", pick(lambda p: len(p["_gt"]["tables"]) == 1, 12)),
        ("formula", "formula", pick(lambda p: len(p["_gt"]["formulas"]) >= 2, 12)),
        ("multi-column", "text", pick(lambda p: p["_attr"]["layout"] in ("double_column", "three_column")
                                      and not p["_gt"]["tables"], 10)),
        ("slides", "text", pick(lambda p: p["_attr"]["data_source"] == "PPT2PDF", 6)),
        ("handwritten-notes", "text", pick(lambda p: p["_attr"]["data_source"] == "note", 6)),
        ("exam-paper", "text", pick(lambda p: p["_attr"]["data_source"] == "exam_paper", 6)),
        ("textbook", "text", pick(lambda p: p["_attr"]["data_source"] in ("colorful_textbook", "book"), 6)),
    ]
    rows = []
    out = DATA / "omnidoc"
    for cat, kind, ps in plan:
        for p in ps:
            name = p["page_info"]["image_path"]
            img = out / "images" / name
            get(f"{HF}/opendatalab/OmniDocBench/resolve/main/images/{urllib.parse.quote(name)}", img)
            g = p["_gt"]
            rows.append({"id": f"omnidoc/{Path(name).stem}", "category": cat, "kind": kind, "lang": "en",
                         "image": f"images/{name}", "gt": g["gt"],
                         "gt_html": g["tables"][0] if kind == "table" else None,
                         "gt_formulas": g["formulas"] if kind == "formula" else None,
                         "source": p["_attr"]["data_source"]})
    write_set("omnidoc", rows)
    return rows


def real5(clean: list[dict]) -> None:
    """Two pages per acquisition condition, drawn from the clean selection so content matches."""
    conds = ["Scanning", "Warping", "Screen-Photography", "Illumination", "Skew"]
    rows = []
    pool = [r for r in clean if r["kind"] != "formula"]
    for cond in conds:
        listing = json.loads(get(f"https://huggingface.co/api/datasets/PaddlePaddle/Real5-OmniDocBench/tree/main/"
                                 f"Real5-OmniDocBench-{cond}"))
        have = {Path(x["path"]).stem: x["path"] for x in listing}
        cands = [r for r in pool if Path(r["image"]).stem in have]
        for r in rng.sample(cands, min(2, len(cands))):
            path = have[Path(r["image"]).stem]
            dest = DATA / "real5" / "images" / Path(path).name
            get(f"{HF}/PaddlePaddle/Real5-OmniDocBench/resolve/main/{urllib.parse.quote(path)}", dest)
            rows.append({**{k: v for k, v in r.items() if k != "set"}, "id": f"real5/{cond}/{dest.stem}",
                         "category": f"degraded-{cond.lower()}", "image": f"images/{dest.name}"})
    write_set("real5", rows)


def lowres(clean: list[dict]) -> None:
    rows = []
    pool = [r for r in clean if r["category"] in ("multi-column", "textbook", "exam-paper", "table")]
    for r in rng.sample(pool, 8):
        im = Image.open(DATA / "omnidoc" / r["image"]).convert("RGB")
        im = im.resize((int(im.width * 0.4), int(im.height * 0.4)), Image.LANCZOS)
        dest = DATA / "lowres" / "images" / (Path(r["image"]).stem + ".jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=70)
        rows.append({**{k: v for k, v in r.items() if k != "set"}, "id": f"lowres/{dest.stem}",
                     "category": "low-resolution", "image": f"images/{dest.name}"})
    write_set("lowres", rows)


# -- Vietnamese pages, typeset from Wikipedia ------------------------------------------
VI_ARTICLES = ["Hà_Nội", "Vịnh_Hạ_Long", "Phở", "Áo_dài", "Sông_Mê_Kông", "Nguyễn_Du"]


def vi_pages() -> None:
    import pypdfium2 as pdfium

    from localocr import publish

    rows = []
    out = DATA / "vi-pages" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for k, title in enumerate(VI_ARTICLES):
        q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "plain",
                                    "titles": title, "format": "json", "redirects": 1})
        pages = json.loads(get(f"https://vi.wikipedia.org/w/api.php?{q}"))["query"]["pages"]
        text = next(iter(pages.values()))["extract"]
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 120][:6]
        body, words = [], 0
        for p in paras:  # keep it to one page
            if words > 330:
                break
            body.append(p)
            words += len(p.split())
        table = ("| Hạng mục | Giá trị | Ghi chú |\n|---|---|---|\n"
                 f"| Năm ghi nhận | {1990 + k * 5} | Theo tài liệu lưu trữ |\n"
                 f"| Số lượng | {1234 + k * 111:,} | Ước tính gần đúng |\n"
                 "| Đánh giá | Tốt | Cần kiểm chứng thêm |")
        md = f"# {title.replace('_', ' ')}\n\n" + "\n\n".join(body[:2]) + f"\n\n{table}\n\n" + "\n\n".join(body[2:])
        pdf_bytes, _ = publish.convert(md, "pdf", {})
        pdf = pdfium.PdfDocument(pdf_bytes)
        if len(pdf) > 1:  # trim the ground truth to what landed on page 1
            first = pdf[0].get_textpage().get_text_range()
            last_words = re.sub(r"\s+", " ", first).strip()[-40:]
            md = md[: md.find(last_words[-25:]) + 25] if last_words[-25:] in md else md
        clean = pdf[0].render(scale=150 / 72).to_pil().convert("RGB")
        pdf.close()
        name = f"vi-{k + 1}"
        clean.save(out / f"{name}-clean.png")
        rows.append({"id": f"vi-pages/{name}-clean", "category": "vietnamese-page", "kind": "text", "lang": "vi",
                     "image": f"images/{name}-clean.png", "gt": md})
        # a phone-photo-ish copy: lower resolution, slight rotation, blur, grey cast, JPEG artefacts
        d = clean.resize((int(clean.width * 0.6), int(clean.height * 0.6)), Image.BILINEAR)
        d = d.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor=(205, 200, 190))
        d = d.filter(ImageFilter.GaussianBlur(0.8))
        d = Image.blend(d, Image.new("RGB", d.size, (190, 180, 160)), 0.18)
        buf = io.BytesIO()
        d.save(buf, "JPEG", quality=35)
        (out / f"{name}-degraded.jpg").write_bytes(buf.getvalue())
        rows.append({"id": f"vi-pages/{name}-degraded", "category": "vietnamese-page-degraded", "kind": "text",
                     "lang": "vi", "image": f"images/{name}-degraded.jpg", "gt": md})
    write_set("vi-pages", rows)


# -- line-level sets (parquet) ------------------------------------------------------------
def parquet_lines(repo: str, path: str, n: int, text_col: str | None, img_col: str | None) -> list[tuple]:
    import pandas as pd

    raw = RAW / repo.replace("/", "__") / Path(path).name
    df = pd.read_parquet(io.BytesIO(get(f"{HF}/{repo}/resolve/main/{path}", raw)))
    text_col = text_col or next(c for c in df.columns if c.lower() in ("text", "label", "transcription", "ground_truth"))
    img_col = img_col or next(c for c in df.columns if "image" in c.lower())
    rows = df.sample(n, random_state=7)
    return [(r[img_col]["bytes"], str(r[text_col])) for _, r in rows.iterrows()]


def vi_lines() -> None:
    out = DATA / "vi-lines" / "images"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for repo, n, prefix in (("HenryBui/vietnamese_handwriting_line_ocr", 12, "hw"),
                            ("manhha2502/Vietnamese_Handwriting_OCR", 6, "hwpara")):
        for k, (b, text) in enumerate(parquet_lines(repo, "data/test-00000-of-00001.parquet", n, "content", "image")):
            name = f"{prefix}-{k:02d}.png"
            Image.open(io.BytesIO(b)).convert("RGB").save(out / name)
            rows.append({"id": f"vi-lines/{Path(name).stem}", "category": "vietnamese-handwriting", "kind": "text",
                         "lang": "vi", "image": f"images/{name}", "gt": text})
    labels = json.loads(get(f"{HF}/ihbkaiser/synthetic-vi-news-ocr/resolve/main/labels.json"))
    for k, item in enumerate(rng.sample(labels, 10)):
        name = f"print-{k:02d}.png"
        rel = item["image_path"].split("/", 1)[1]
        get(f"{HF}/ihbkaiser/synthetic-vi-news-ocr/resolve/main/{rel}", out / name)
        rows.append({"id": f"vi-lines/{Path(name).stem}", "category": "vietnamese-printed-line", "kind": "text",
                     "lang": "vi", "image": f"images/{name}", "gt": item["label"]})
    write_set("vi-lines", rows)


def iam() -> None:
    out = DATA / "iam" / "images"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, (b, text) in enumerate(parquet_lines("Teklia/IAM-line", "data/test.parquet", 12, "text", "image")):
        name = f"line-{k:02d}.png"
        Image.open(io.BytesIO(b)).convert("RGB").save(out / name)
        rows.append({"id": f"iam/{Path(name).stem}", "category": "english-handwriting", "kind": "text",
                     "lang": "en", "image": f"images/{name}", "gt": text})
    write_set("iam", rows)


if __name__ == "__main__":
    import sys

    only = sys.argv[1:]
    if not only or {"omnidoc", "real5", "lowres"} & set(only):
        clean = omnidoc()
        real5(clean)
        lowres(clean)
    for name, fn in (("vi-pages", vi_pages), ("vi-lines", vi_lines), ("iam", iam)):
        if not only or name in only:
            fn()
