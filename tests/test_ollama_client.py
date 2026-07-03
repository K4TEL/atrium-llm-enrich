"""
tests/test_ollama_client.py
============================
Tests for ollama_client.py, focused on the 2026-07-03 review-pass fix for
finding #2: ensure_model_pulled()'s bare-name tag matching over-matched —
it treated e.g. 'qwen2.5:14b' as satisfying a request for 'qwen2.5:7b',
skipping the pull and then running inference against a tag Ollama never
had (and could crash outright on a model entry missing its 'name' key).
The fix requires an EXACT tag match, with a bare (untagged) request
additionally satisfied only by '<name>:latest'.
"""

import json
from unittest.mock import MagicMock

import pytest
import requests

from ollama_client import ensure_model_pulled

# ── ensure_model_pulled ──────────────────────────────────────────────────────


def _session_with_tags(names):
    """A MagicMock stand-in for requests.Session whose GET /api/tags returns
    the given model names."""
    session = MagicMock()
    tags_resp = MagicMock()
    tags_resp.raise_for_status.return_value = None
    tags_resp.json.return_value = {"models": [{"name": n} for n in names]}
    session.get.return_value = tags_resp
    return session


def test_ensure_model_pulled_exact_tag_present_skips_pull():
    session = _session_with_tags(["qwen2.5:7b"])
    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
    session.post.assert_not_called()


def test_ensure_model_pulled_bare_name_satisfied_by_latest():
    session = _session_with_tags(["qwen2.5:latest"])
    ensure_model_pulled("http://x", "qwen2.5", session, timeout=10)
    session.post.assert_not_called()


def test_ensure_model_pulled_different_tag_of_same_family_still_pulls():
    # THE bug this fixes: qwen2.5:14b being installed must NOT satisfy a
    # request for qwen2.5:7b — the old bare-name-family match skipped the
    # pull here and then ran inference against a tag Ollama didn't have.
    session = _session_with_tags(["qwen2.5:14b"])
    session.post.return_value.__enter__.return_value = MagicMock(
        iter_lines=lambda: [], raise_for_status=lambda: None
    )
    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"] == {"name": "qwen2.5:7b"}


def test_ensure_model_pulled_missing_model_streams_pull_progress(capsys):
    session = _session_with_tags([])
    pull_resp = MagicMock()
    pull_resp.raise_for_status.return_value = None
    pull_resp.iter_lines.return_value = [
        json.dumps({"status": "pulling manifest"}).encode(),
        json.dumps({"status": "success"}).encode(),
    ]
    session.post.return_value.__enter__.return_value = pull_resp

    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)

    out = capsys.readouterr().out
    assert "pulling manifest" in out
    assert "success" in out
    assert "ready" in out


def test_ensure_model_pulled_ignores_tags_entries_without_a_name():
    # A registry row with no 'name' key (e.g. a manifest-only entry) must not
    # crash the set comprehension, and must not itself count as a match.
    session = MagicMock()
    tags_resp = MagicMock()
    tags_resp.raise_for_status.return_value = None
    tags_resp.json.return_value = {"models": [{"digest": "sha256:abc"}, {"name": "qwen2.5:7b"}]}
    session.get.return_value = tags_resp

    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
    session.post.assert_not_called()


def test_ensure_model_pulled_unreachable_host_raises_runtime_error():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("refused")

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
