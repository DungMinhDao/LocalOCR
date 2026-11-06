"""Accuracy metrics for OCR output against ground truth (OmniDocBench-style, simplified).

- text:    1 − normalised edit distance after stripping Markdown/HTML formatting (reading order counts).
- table:   TEDS (tree edit distance similarity) over the HTML table structure + cell text.
- formula: 1 − normalised edit distance of whitespace-free LaTeX.
"""

import html as htmllib
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# -- normalisation ---------------------------------------------------------------
_MD_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", re.M)


def md_tables_to_html(md: str) -> str:
    """Turn pipe tables into <table> so text and TEDS treat Markdown and HTML tables alike."""
    out, block = [], []

    def flush():
        if len(block) >= 2 and _MD_TABLE_RULE.match(block[1]):
            rows = [block[0]] + block[2:]
            cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
            out.append("<table>" + "".join("<tr>" + "".join(f"<td>{htmllib.escape(c)}</td>" for c in r) + "</tr>"
                                           for r in cells) + "</table>")
        else:
            out.extend(block)
        block.clear()

    for line in md.splitlines():
        if line.strip().startswith("|") or (block and "|" in line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def plain(text: str) -> str:
    """Formatting-free text: what a reader would see, for edit distance."""
    t = unicodedata.normalize("NFC", text or "")
    t = md_tables_to_html(t)
    t = re.sub(r"<(td|th)[^>]*>", " ", t, flags=re.I)
    t = re.sub(r"<br\s*/?>|</(p|div|tr|li|h\d)>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)                       # remaining tags
    t = htmllib.unescape(t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)           # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)       # links → text
    t = re.sub(r"^\s{0,3}(#{1,6}|[-*+]|\d+[.)])\s+", "", t, flags=re.M)   # headings, bullets
    t = re.sub(r"(\*\*|__|\*|_|`|~~)", "", t)
    t = re.sub(r"^\s*[-=*]{3,}\s*$", "", t, flags=re.M)
    t = re.sub(r"\$\$?|\\\(|\\\)|\\\[|\\\]", "", t)      # math delimiters
    t = t.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", t).strip()


def text_score(pred: str, gt: str) -> float:
    a, b = plain(pred), plain(gt)
    if not a and not b:
        return 1.0
    return 1 - Levenshtein.normalized_distance(a, b)


def latex_norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\$\$?|\$\$?$|^\\\[|\\\]$|^\\\(|\\\)$", "", s.strip()).strip()
    s = re.sub(r"\\(left|right|displaystyle|textstyle|mathrm\s*(?=\{))", "", s)
    s = re.sub(r"\\[,;:!> ]|~", "", s)
    return re.sub(r"\s+", "", s)


def formula_score(pred: str, gt: str) -> float:
    a, b = latex_norm(pred), latex_norm(gt)
    if not a and not b:
        return 1.0
    return 1 - Levenshtein.normalized_distance(a, b)


# -- TEDS --------------------------------------------------------------------------
def _table_tree(html_or_md: str):
    from lxml import html as lh

    src = md_tables_to_html(html_or_md or "")
    m = re.search(r"<table.*?</table>", src, flags=re.S | re.I)
    if not m:
        return None
    try:
        return lh.fromstring(m.group(0))
    except Exception:
        return None


def teds(pred: str, gt: str) -> float:
    """Tree edit distance similarity (Zhong et al. 2020) over tr/td structure with cell text."""
    from apted import APTED, Config

    class Node:
        def __init__(self, tag, colspan=1, rowspan=1, text=""):
            self.tag, self.colspan, self.rowspan, self.text, self.children = tag, colspan, rowspan, text, []

    class Cfg(Config):
        def rename(self, a, b):
            if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
                return 1.0
            if a.tag == "td":
                return Levenshtein.normalized_distance(a.text, b.text)
            return 0.0

        def children(self, n):
            return n.children

    def build(el):
        tag = "td" if el.tag in ("td", "th") else el.tag
        n = Node(tag, int(el.get("colspan", 1) or 1), int(el.get("rowspan", 1) or 1),
                 re.sub(r"\s+", " ", el.text_content()).strip() if tag == "td" else "")
        if tag != "td":
            for c in el:
                if c.tag in ("tr", "td", "th", "thead", "tbody"):
                    if c.tag in ("thead", "tbody"):   # flatten sections: different models emit them differently
                        for r in c:
                            n.children.append(build(r))
                    else:
                        n.children.append(build(c))
        return n

    def count(n):
        return 1 + sum(count(c) for c in n.children)

    gt_el = _table_tree(gt)
    pr_el = _table_tree(pred)
    if gt_el is None:
        return float("nan")
    if pr_el is None:
        return 0.0
    a, b = build(pr_el), build(gt_el)
    dist = APTED(a, b, Cfg()).compute_edit_distance()
    return max(0.0, 1 - dist / max(count(a), count(b)))


_DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\begin\{(equation|align|gather)\*?\}(.+?)\\end\{\3\*?\}", re.S)


def formula_page_score(pred: str, formulas: list[str]) -> float:
    """Each ground-truth display formula against its best-matching display formula in the output."""
    found = [next(g for g in m.groups() if g and g not in ("equation", "align", "gather"))
             for m in _DISPLAY_MATH.finditer(pred or "")]
    if not formulas:
        return float("nan")
    if not found:
        return 0.0
    return sum(max(formula_score(f, gt) for f in found) for gt in formulas) / len(formulas)
