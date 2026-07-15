"""
tests/test_eval_metrics.py – Unit tests for eval_metrics.py, the Document
Understanding benchmark scoring primitives (hub issue #22): CER/WER, normalized
edit distance, entity F1, page/corpus aggregation, and the optional TEDS guard.

All deterministic and pure-stdlib.
"""

import pytest

from eval_metrics import (
    cer,
    entity_prf,
    levenshtein,
    normalize_text,
    normalized_edit_distance,
    score_corpus,
    score_page,
    teds_score,
    wer,
)


# ── normalize_text ──────────────────────────────────────────────────────────
def test_normalize_collapses_whitespace():
    assert normalize_text("a   b\tc\n") == "a b c"


def test_normalize_lowercase_is_optional():
    assert normalize_text("ABC", lowercase=True) == "abc"
    assert normalize_text("ABC") == "ABC"


# ── levenshtein ─────────────────────────────────────────────────────────────
def test_levenshtein_characters():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("abc", "") == 3
    assert levenshtein("abc", "abc") == 0


def test_levenshtein_word_lists():
    assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1


# ── cer / wer ───────────────────────────────────────────────────────────────
def test_cer_identical_is_zero():
    assert cer("hello", "hello") == 0.0


def test_cer_empty_ref_cases():
    assert cer("", "") == 0.0
    assert cer("", "abc") == 1.0


def test_cer_single_substitution():
    assert cer("abcd", "abxd") == pytest.approx(0.25)


def test_wer_word_level():
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_empty_ref_cases():
    assert wer("", "") == 0.0
    assert wer("", "a b") == 1.0


# ── normalized_edit_distance ────────────────────────────────────────────────
def test_ned_both_empty_is_zero():
    assert normalized_edit_distance("", "") == 0.0


def test_ned_uses_max_length_denominator():
    # ref len 3, hyp len 5, distance 2 → 2 / max(3, 5) = 0.4
    assert normalized_edit_distance("abc", "abcde") == pytest.approx(0.4)


# ── entity_prf ──────────────────────────────────────────────────────────────
def test_entity_prf_perfect_match():
    ents = [("PER", "Jan"), ("LOC", "Praha")]
    r = entity_prf(ents, ents)
    assert (r["precision"], r["recall"], r["f1"]) == (1.0, 1.0, 1.0)
    assert r["tp"] == 2


def test_entity_prf_partial_overlap():
    ref = [("PER", "Jan"), ("LOC", "Praha")]
    hyp = [("PER", "Jan"), ("LOC", "Brno")]
    r = entity_prf(ref, hyp)
    assert r["tp"] == 1
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)


def test_entity_prf_empty_inputs():
    r = entity_prf([], [])
    assert (r["precision"], r["recall"], r["f1"]) == (0.0, 0.0, 0.0)


def test_entity_prf_text_is_case_insensitive():
    assert entity_prf([("PER", "Jan")], [("PER", "jan")])["tp"] == 1


# ── score_page / score_corpus ───────────────────────────────────────────────
def test_score_page_reports_all_metrics():
    p = score_page("hello world", "hello world")
    assert p["cer"] == 0.0 and p["wer"] == 0.0 and p["ned"] == 0.0
    assert p["ref_chars"] == len("hello world")


def test_score_corpus_macro_averages_with_overall():
    pages = [
        {"cer": 0.0, "wer": 0.0, "ned": 0.0},
        {"cer": 0.4, "wer": 0.5, "ned": 0.2},
    ]
    out = score_corpus(pages, ["clean", "hard"])
    assert out["clean"]["cer"] == 0.0
    assert out["hard"]["cer"] == pytest.approx(0.4)
    assert out["overall"]["cer"] == pytest.approx(0.2)
    assert out["overall"]["n_pages"] == 2.0


def test_score_corpus_length_mismatch_raises():
    with pytest.raises(ValueError):
        score_corpus([{"cer": 0, "wer": 0, "ned": 0}], ["a", "b"])


# ── teds_score optional dependency ──────────────────────────────────────────
def test_teds_score_without_optional_dep_raises_importerror():
    with pytest.raises(ImportError):
        teds_score("<table></table>", "<table></table>")
