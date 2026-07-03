"""
tests/test_openrouter_client.py
================================
Tests for openrouter_client.py, focused on the 2026-07-03 review-pass fix
for finding #1: --attach-as-file was dead code. _build_attachment_content()
existed and was documented but run_document_level() had no way to receive
it, so the document body was always inlined as plain text and the
file-attachment path did nothing regardless of the flag. These tests
exercise _build_attachment_content() directly, then reproduce the exact
per-document closure _make_doc_builder() builds inside main() to confirm
its output is what actually reaches the chat_fn once wired through
llm_client_shared.run_document_level()'s user_content_builder parameter.
"""

import base64
import json

from llm_client_shared import run_document_level
from openrouter_client import _build_attachment_content, build_arg_parser

# ── _build_attachment_content ───────────────────────────────────────────────


def test_build_attachment_content_inlines_by_default():
    content = _build_attachment_content("Vyzkum odhalil zaklady.", "sample.md", as_file=False)
    assert content == "DOCUMENT:\nVyzkum odhalil zaklady."


def test_build_attachment_content_as_file_encodes_base64_file_part():
    content = _build_attachment_content("Vyzkum odhalil zaklady.", "sample.md", as_file=True)
    assert isinstance(content, list)
    text_part, file_part = content
    assert text_part == {"type": "text", "text": "DOCUMENT (attached below):"}
    assert file_part["type"] == "file"
    assert file_part["file"]["filename"] == "sample.md"

    data_url = file_part["file"]["file_data"]
    assert data_url.startswith("data:text/markdown;base64,")
    b64_payload = data_url.split(",", 1)[1]
    assert base64.b64decode(b64_payload).decode("utf-8") == "Vyzkum odhalil zaklady."


# ── run_document_level(user_content_builder=...) — finding #1 regression ───


def _fake_empty_items_chat_fn(captured):
    def chat_fn(messages):
        captured.append(messages)
        return json.dumps({"items": []})

    return chat_fn


class _DummyDocModel:
    """Minimal stand-in for build_document_schema()'s DocumentEnrichment —
    only .items is ever read by run_document_level()."""

    items = []

    @classmethod
    def model_validate_json(cls, data):
        return cls()


def test_attach_as_file_reaches_run_document_level_when_wired(tmp_path):
    doc_path = tmp_path / "sample.md"
    doc_path.write_text("Vyzkum odhalil zaklady.", encoding="utf-8")

    captured = []

    def doc_builder(doc_text):
        # exactly the closure _make_doc_builder() returns inside main()
        return _build_attachment_content(doc_text, doc_path.name, True)

    run_document_level(
        doc_path,
        _fake_empty_items_chat_fn(captured),
        "system prompt",
        _DummyDocModel,
        user_content_builder=doc_builder,
    )

    sent_content = captured[0][1]["content"]
    assert isinstance(sent_content, list)  # the file-attachment content part, not inlined text
    assert sent_content[1]["file"]["filename"] == "sample.md"


def test_attach_as_file_flag_defaults_to_false():
    args = build_arg_parser().parse_args([])
    assert args.attach_as_file is False

    args = build_arg_parser().parse_args(["--attach-as-file"])
    assert args.attach_as_file is True
