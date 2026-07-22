"""
api_util/pdf_to_md.py — digital-born PDF → visually-rich Markdown.

Converts a PDF that already carries an extractable text layer into the
page-sectioned Markdown consumed by ``run_document_level()``, enriched with the
visual-layout cues from ``layout_md.py`` (issue #10): per-page ``DOC_META``
(canvas size), ``## Page N`` + ``PAGE_BREAK`` boundaries, per-block ``BBOX`` and
``FONT`` cues, and GFM tables.

Scope (per issue #10, first pass): **digital-born only**. Curve-only and scanned
pages have no trustworthy text layer, so this converter marks them with a
``NEEDS_OCR`` cue and moves on — the render+OCR path (Tesseract/PERO) is a
benchmark-gated follow-up (hub ``atrium-project#22``). A born-digital page whose
subset fonts lack a ``/ToUnicode`` map extracts *garbled* text; a lightweight
**decode-sanity check** flags those as ``NEEDS_OCR`` too rather than trusting them.

Uses **pdfplumber** (MIT: word/char bounding boxes, font names/sizes, tables),
imported lazily so the base install never requires it — mirroring
``flexiconv_convert.py``'s ``*_available()`` pattern.
"""

from __future__ import annotations

import argparse
import logging
import re
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util import layout_md as L  # noqa: E402
from api_util.teitok_read import doc_id_from_path  # noqa: E402

INSTALL_HINT = "pdfplumber is not installed. Please run: pip install -r requirements_docmd.txt"

# A page with fewer real characters than this has no usable text layer.
MIN_TEXT_CHARS = 3
# Fraction of suspicious codepoints above which an extracted layer is "garbled".
GARBLE_THRESHOLD = 0.15
# Subset-font tag prefix, e.g. "ABCDEF+TimesNewRoman".
_SUBSET_RE = re.compile(r"^[A-Z]{6}\+")


class PdfPlumberNotInstalled(RuntimeError):
    pass


def pdfplumber_available() -> bool:
    """Whether pdfplumber can be imported, without raising."""
    try:
        import pdfplumber  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Text-layer quality
# --------------------------------------------------------------------------- #
def _looks_garbled(text: str) -> bool:
    """Heuristic decode-sanity check on an extracted text layer.

    Flags a high fraction of Unicode replacement chars (``U+FFFD``), C0/C1
    control chars, and private-use-area codepoints — the fingerprint of a subset
    font decoded without a ``/ToUnicode`` map. NOTE: it cannot catch the subtler
    case where diacritics decode to *valid but wrong* Latin letters (Czech
    ``ě→I``); that needs a dictionary/LM hit-rate check and is left to the OCR
    follow-up (issue #10 / hub #22).
    """
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return False
    bad = 0
    for c in stripped:
        if c == "�":
            bad += 1
            continue
        cat = unicodedata.category(c)
        if cat in {"Cc", "Cf", "Co", "Cn"}:  # control / format / private-use / unassigned
            bad += 1
    return bad / len(stripped) > GARBLE_THRESHOLD


def _page_ocr_reason(text: str) -> Optional[str]:
    """Return an OCR reason if this page's text layer is unusable, else None."""
    if len(text.strip()) < MIN_TEXT_CHARS:
        return "no extractable text layer"
    if _looks_garbled(text):
        return "garbled text layer (subset font without /ToUnicode?)"
    return None


# --------------------------------------------------------------------------- #
# Font / block helpers
# --------------------------------------------------------------------------- #
def _clean_fontname(name: Optional[str]) -> Optional[str]:
    """Strip a subset tag from a font name: ``ABCDEF+Times`` -> ``Times``."""
    if not name:
        return None
    return _SUBSET_RE.sub("", name)


def _dominant_font(chars: List[dict]) -> str:
    """FONT cue for the most common (family, size) among a block's chars."""
    families = Counter(_clean_fontname(c.get("fontname")) for c in chars if c.get("fontname"))
    sizes = [round(float(c["size"]), 1) for c in chars if c.get("size") is not None]
    family = families.most_common(1)[0][0] if families else None
    size = Counter(sizes).most_common(1)[0][0] if sizes else None
    return L.font(size=size, family=family)


def _in_any_table(line: dict, table_bboxes: List[tuple]) -> bool:
    """True if a text line's vertical midpoint sits inside a table region."""
    mid_y = (line["top"] + line["bottom"]) / 2
    for x0, top, x1, bottom in table_bboxes:
        if top <= mid_y <= bottom and line["x1"] > x0 and line["x0"] < x1:
            return True
    return False


def _line_size(line: dict) -> Optional[float]:
    """Dominant rounded font size among a line's chars, or None."""
    sizes = [round(float(c["size"]), 1) for c in line.get("chars", []) if c.get("size") is not None]
    return Counter(sizes).most_common(1)[0][0] if sizes else None


def _group_lines_into_blocks(lines: List[dict]) -> List[List[dict]]:
    """Group consecutive text lines into blocks.

    A block break is inserted on a vertical gap larger than the typical line
    height, or on a meaningful font-size change between lines — so a large
    heading line becomes its own block and keeps its ``FONT`` size cue instead
    of being flattened into the surrounding body text.
    """
    if not lines:
        return []
    lines = sorted(lines, key=lambda ln: ln["top"])
    heights = [ln["bottom"] - ln["top"] for ln in lines if ln["bottom"] > ln["top"]]
    median_h = statistics.median(heights) if heights else 12.0
    gap_threshold = max(median_h * 1.4, 3.0)

    blocks: List[List[dict]] = [[lines[0]]]
    for prev, cur in zip(lines, lines[1:], strict=False):
        prev_size, cur_size = _line_size(prev), _line_size(cur)
        size_changed = (
            prev_size is not None and cur_size is not None and abs(cur_size - prev_size) >= 1.5
        )
        if cur["top"] - prev["bottom"] > gap_threshold or size_changed:
            blocks.append([cur])
        else:
            blocks[-1].append(cur)
    return blocks


def _block_bbox(block: List[dict]) -> List[float]:
    """[x_min, y_min, x_max, y_max] covering every line in a block."""
    return [
        min(ln["x0"] for ln in block),
        min(ln["top"] for ln in block),
        max(ln["x1"] for ln in block),
        max(ln["bottom"] for ln in block),
    ]


def _block_md(block: List[dict]) -> str:
    """Render a text block: BBOX + FONT cues then the reflowed block text."""
    text = " ".join(ln["text"].strip() for ln in block if ln.get("text", "").strip()).strip()
    if not text:
        return ""
    chars = [c for ln in block for c in ln.get("chars", [])]
    cues = L.bbox(_block_bbox(block))
    font_cue = _dominant_font(chars)
    header = f"{cues} {font_cue}".strip() if font_cue else cues
    return f"{header}\n{text}"


# --------------------------------------------------------------------------- #
# Per-page rendering
# --------------------------------------------------------------------------- #
def _render_page(page, page_num: int) -> List[str]:
    """Render a single pdfplumber page to a list of Markdown parts."""
    parts: List[str] = []
    if page_num > 1:
        parts.append(L.page_break(page_num))
    parts.append(f"\n## Page {page_num}\n")

    orient = "portrait" if page.height >= page.width else "landscape"
    parts.append(L.doc_meta(size=f"{round(page.width)}x{round(page.height)}pt", orientation=orient))

    text = page.extract_text() or ""
    ocr_reason = _page_ocr_reason(text)
    if ocr_reason:
        parts.append(L.needs_ocr(page_num, ocr_reason))
        return parts

    # Tables first, so their regions can be excluded from flowing text.
    table_items = []
    table_bboxes = []
    try:
        for table in page.find_tables():
            rendered = L.md_table(table.extract(), header=True)
            if rendered:
                table_items.append((table.bbox[1], rendered, list(table.bbox)))
                table_bboxes.append(table.bbox)
    except Exception as exc:
        logger.warning("Table detection failed on page %d; tables skipped: %s", page_num, exc)

    try:
        lines = page.extract_text_lines(strip=True)
    except Exception as exc:
        logger.warning("Text-line extraction failed on page %d; page will be empty: %s", page_num, exc)
        lines = []
    lines = [ln for ln in lines if not _in_any_table(ln, table_bboxes)]

    # Interleave text blocks and tables in reading (top-to-bottom) order.
    items = []
    for block in _group_lines_into_blocks(lines):
        rendered = _block_md(block)
        if rendered:
            items.append((_block_bbox(block)[1], rendered))
    for top, rendered, tbbox in table_items:
        items.append((top, f"{L.bbox(tbbox)}\n{rendered}"))

    for _top, rendered in sorted(items, key=lambda it: it[0]):
        parts.append(rendered)
    return parts


def convert(path: str | Path) -> str:
    """Convert a digital-born ``.pdf`` to visually-rich, page-sectioned Markdown."""
    if not pdfplumber_available():
        raise PdfPlumberNotInstalled(INSTALL_HINT) from None
    import pdfplumber

    path = Path(path)
    parts: List[str] = [f"# {doc_id_from_path(path)}"]
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            parts.extend(_render_page(page, i))
    return "\n".join(parts).strip() + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None, help="Write to file instead of stdout."
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        rendered = convert(args.input_file)
    except PdfPlumberNotInstalled as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"-> {args.output}")
    else:
        print(rendered)
