"""
tests/test_pdf_to_md.py
=======================
Tests for api_util/pdf_to_md.py — digital-born PDF → visually-rich Markdown.

Minimal PDFs are hand-built in-process (no committed binaries, no reportlab):
one with a real text layer (digital-born) and one with none (the scanned/curve
class). The decode-sanity helpers are pure and always tested; the end-to-end
conversion tests are skipped cleanly where pdfplumber isn't installed.
"""

import pytest

from api_util import pdf_to_md
from api_util.pdf_to_md import _looks_garbled, _page_ocr_reason

requires_pdfplumber = pytest.mark.skipif(
    not pdf_to_md.pdfplumber_available(), reason="pdfplumber not installed"
)


# --------------------------------------------------------------------------- #
# Hermetic minimal-PDF builder
# --------------------------------------------------------------------------- #
def _write_min_pdf(path, content_stream: str, with_font: bool = True) -> None:
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << %s >> /Contents 4 0 R >>"
        % (b"/Font << /F1 5 0 R >>" if with_font else b""),
    ]
    stream = content_stream.encode("latin-1")
    objs.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (i, body)
    xref_pos = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        len(objs) + 1,
        xref_pos,
    )
    path.write_bytes(out)


@pytest.fixture
def digital_pdf(tmp_path):
    content = (
        "BT /F1 18 Tf 72 720 Td (Vyzkum lokality) Tj ET\n"
        "BT /F1 12 Tf 72 690 Td (Sonda odhalila zaklady kostela.) Tj ET\n"
        "BT /F1 12 Tf 72 672 Td (Druhy radek textu na strance.) Tj ET"
    )
    dest = tmp_path / "CTX100.pdf"
    _write_min_pdf(dest, content, with_font=True)
    return dest


@pytest.fixture
def scan_pdf(tmp_path):
    dest = tmp_path / "CTX200.pdf"
    _write_min_pdf(dest, "0 0 0 RG 72 72 400 200 re S", with_font=False)
    return dest


# --------------------------------------------------------------------------- #
# Decode-sanity helpers (pure — always run)
# --------------------------------------------------------------------------- #
def test_looks_garbled_flags_replacement_and_control_chars():
    assert _looks_garbled("���� normal") is True
    assert _looks_garbled("Sonda odhalila zaklady kostela.") is False
    assert _looks_garbled("   ") is False  # empty/whitespace is not "garbled"


def test_page_ocr_reason_thresholds():
    assert _page_ocr_reason("") == "no extractable text layer"
    assert _page_ocr_reason("ab") == "no extractable text layer"
    assert "garbled" in (_page_ocr_reason("������") or "")
    assert _page_ocr_reason("A clean sentence of real text.") is None


# --------------------------------------------------------------------------- #
# End-to-end conversion (needs pdfplumber)
# --------------------------------------------------------------------------- #
@requires_pdfplumber
def test_convert_digital_born_emits_cues(digital_pdf):
    md = pdf_to_md.convert(digital_pdf)
    assert md.startswith("# CTX100")
    assert "## Page 1" in md
    assert "<!-- DOC_META: size=612x792pt, orientation=portrait -->" in md
    assert "<!-- BBOX: [" in md
    assert "<!-- FONT: size=18pt" in md  # heading kept its own size cue
    assert "Vyzkum lokality" in md
    assert "Sonda odhalila zaklady kostela." in md


@requires_pdfplumber
def test_convert_textless_page_marks_needs_ocr(scan_pdf):
    md = pdf_to_md.convert(scan_pdf)
    assert "## Page 1" in md
    assert "<!-- NEEDS_OCR: pg_1 (no extractable text layer) -->" in md
    # a text-less page emits no fabricated body text
    assert "BBOX" not in md


@requires_pdfplumber
def test_convert_missing_lib_raises(monkeypatch, digital_pdf):
    monkeypatch.setattr(pdf_to_md, "pdfplumber_available", lambda: False)
    with pytest.raises(pdf_to_md.PdfPlumberNotInstalled):
        pdf_to_md.convert(digital_pdf)
