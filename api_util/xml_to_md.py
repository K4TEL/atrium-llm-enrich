"""
api_util/xml_to_md.py — TEITOK / ALTO → Markdown converter.

Renders a whole XML document (TEITOK ``*.teitok.xml`` or raw ALTO) to
Markdown or plain text, so entire documents can be fed to LLMs as a single
prompt (local or as an OpenRouter file/text attachment) — document-level
input, complementing the existing line-level CSV/TEITOK row reader in
llm_client_shared.read_input_rows() / llm_utils.read_input_rows().

Builds on teitok_read.py (TEITOK) and a small, dependency-free ALTO reader
below, following teitok_read.read_teitok_rows()'s row shape
({"page_num", "line_num", "text"}) so both formats feed the same renderer.

Note: teitok_alto.py's ``_parse_alto`` is intentionally NOT reused here — it
is module-private, tightly coupled to the CoNLL-U+NER merge pipeline
(``write_teitok_merged``), and returns bbox/image metadata this converter
doesn't need. ``_read_alto_rows`` below extracts only String/TextLine text,
mirroring the namespace-agnostic parsing style already used in that module.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util.teitok_read import doc_id_from_path, read_teitok_rows  # noqa: E402


def _local_tag(elem: ET.Element) -> str:
    """Strip an XML namespace URI from a tag, e.g. '{ns}Page' -> 'Page'."""
    return elem.tag.split("}")[-1]


def is_alto(path: str | Path) -> bool:
    """Best-effort ALTO detection: peek at the root tag/namespace."""
    try:
        for _event, elem in ET.iterparse(str(path), events=("start",)):
            return _local_tag(elem).lower() == "alto" or "alto" in elem.tag.lower()
    except ET.ParseError:
        return False
    return False


def _read_alto_rows(path: str | Path) -> List[dict]:
    """
    Parses raw ALTO XML (Page > PrintSpace > TextBlock > TextLine > String).
    Returns: list of dicts [{"page_num": int, "line_num": int, "text": str}],
    matching teitok_read.read_teitok_rows()'s row shape.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    rows: List[dict] = []

    page_num = 0
    for page_elem in root.iter():
        if _local_tag(page_elem) != "Page":
            continue
        page_num += 1
        try:
            page_num = int(page_elem.get("PHYSICAL_IMG_NR", page_num))
        except (TypeError, ValueError):
            pass

        line_num = 0
        for line_elem in page_elem.iter():
            if _local_tag(line_elem) != "TextLine":
                continue
            line_num += 1

            words = [
                s.get("CONTENT", "")
                for s in line_elem.iter()
                if _local_tag(s) == "String" and s.get("CONTENT")
            ]
            text = " ".join(w for w in words if w).strip()
            if text:
                rows.append({"page_num": page_num, "line_num": line_num, "text": text})

    return rows


def read_document_rows(path: str | Path) -> List[dict]:
    """Reads rows from either a TEITOK or a raw ALTO XML document."""
    path = Path(path)
    if path.name.lower().endswith(".teitok.xml"):
        return read_teitok_rows(path)
    if is_alto(path):
        return _read_alto_rows(path)
    # Fall back to TEITOK parsing — read_teitok_rows() is namespace-agnostic
    # and will simply return [] if the structure doesn't match, rather than
    # raising, so this is a safe default rather than a silent misdetection.
    return read_teitok_rows(path)


def rows_to_markdown(rows: List[dict], title: str = "") -> str:
    """Renders {page_num, line_num, text} rows as page-sectioned Markdown."""
    if not rows:
        return f"# {title}\n" if title else ""

    parts: List[str] = [f"# {title}"] if title else []
    current_page = None
    for row in rows:
        page = row.get("page_num")
        if page != current_page:
            parts.append(f"\n## Page {page}\n")
            current_page = page
        text = str(row.get("text", "")).strip()
        if text:
            parts.append(text)

    return "\n".join(parts).strip() + "\n"


def rows_to_plain_text(rows: List[dict]) -> str:
    """Renders rows as plain text, one line per row, page breaks as blank lines."""
    parts: List[str] = []
    current_page = None
    for row in rows:
        page = row.get("page_num")
        if current_page is not None and page != current_page:
            parts.append("")
        current_page = page
        text = str(row.get("text", "")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip() + "\n"


def convert(path: str | Path, fmt: str = "markdown") -> str:
    """Convert a TEITOK/ALTO XML document to 'markdown' or 'text'."""
    path = Path(path)
    rows = read_document_rows(path)
    if fmt == "text":
        return rows_to_plain_text(rows)
    return rows_to_markdown(rows, title=doc_id_from_path(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--format", choices=["markdown", "text"], default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="Write to file instead of stdout.")
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    rendered = convert(args.input_file, fmt=args.format)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"-> {args.output}")
    else:
        print(rendered)
