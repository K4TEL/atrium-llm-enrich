# 📓 atrium-llm-enrich — agent_dev_logs/DEVLOG.md (timeline index)
> _LLM-driven enrichment of archaeological documents (local multi-GPU + remote-as-a-service). 3 open
> issues (#8, #10, #11). `test` HEAD `3e7a909` (2026-07-22) · **v0.2.0**._
> _Per-issue detail: `digests/{8,10,11}.digest.md` · `plans/{8,10}.plan.md` · `issues/` exports
> (source of truth). Cross-repo/hub history (DU benchmark #22, spin-out #24) lives in
> `ufal/atrium-project/agent_dev_logs/DEVLOG.md` (deduplicated out of this file)._

## 2026-07-12
- **#8 LLM applications to data — initialization of repository** — Opened by K4TEL as the repo-side
  continuation of hub [ufal/atrium-project#24](https://github.com/ufal/atrium-project/issues/24):
  spin the LLM-only subtasks out of the NameTag3/UDPipe-entangled `atrium-nlp-enrich` into a focused
  sibling repo. Engine copied byte-identical; the actual new work is `openrouter_client.py`,
  `ollama_client.py`, `api_util/xml_to_md.py`, and the torch-free `llm_client_shared.py`.
- **#8 (repo)** — **Document-Understanding scripts** landed (`f2ec956`): `eval_metrics.py` (CER/WER,
  normalized edit distance, entity F1, optional TEDS) and `sample_stratify.py` (quality-stratified
  80/10/10 page sampling) — the hub **#22** benchmark primitives. Licenses test `tests/test_para_licenses.py`
  (`83d7480`), GHA version bumps, dependabot merges (transformers ≥4.57.6, pydantic 2.13.4). Issue #8
  logs + digest/plan added and renamed from the `24.*` hub pair (`1b94264`, `c259453`). Suite at **83
  `def test_`** across 5 files; ruff clean.

## 2026-07-13 → 2026-07-15
- **#8 (repo)** — Test infrastructure hardened: test reqs + formatting (`3779460`); fixtures and tests
  imported from `nlp-enrich` with the `pandas` dependency (`2a09efe`, `43259a4`) — the suite grows well
  past the 83 baseline; `pytest` requirement bumped `>=8.0 → >=9.1.1` (PR #9, `fde9334`); version bump
  (`2540ad9`).

## 2026-07-16
- **#8 / hub #22 (repo)** — DU next steps and test-coverage/reqs updates (`9267382`, `dd90b13`); issue
  logs refreshed (`19e6a03`).

## 2026-07-17
- **#10 PDF and DOCX inputs handling for DU** — Opened by K4TEL: process input PDFs of three kinds
  (curves-instead-of-letters / scanned images / digital-born-with-text), add DOCX as an alternative
  input, and consider DOCX/HTML as intermediate formats on the PDF→LLM path.
- **#11 Decide on inputs of LLMs based on benchmarks existing for DU** — Opened by K4TEL (`question`,
  `development`): a format × detail-level matrix (PDF / InDesign XML / HTML+CSS / DOCX / Page-ALTO XML
  / MD / TXT); find the format used in DU benchmarks toward a FAIR standard; consider
  `ufal/atrium-page-classification` and **Grobid** for PDF processing.
- **#8 (repo)** — Large-model-on-CPU run notes updated (`90a8762`).

## 2026-07-19 → 2026-07-20
- **#10** — **Research pass** (survey + routing, no code): three PDF classes collapse to two paths via
  a cheap `pdffonts`/PyMuPDF font+char+image census (digital-born → extract; curves+scans → render+OCR),
  with a decode-sanity guard for the garbled-diacritics (no-`/ToUnicode`) case. Permissive-first tool
  survey; recommendation to reuse the repo's **Markdown** (doc-level) or **TEITOK** (line-level) targets,
  not a new format. Digest+plan added (`5ee844f`). **Blocker** noted: flexiconv's license is undeclared
  upstream (`para_config.txt:32-33`).
- **#11** — Deep-research report (Gemini) on agentic, roadmap-driven FAIR document navigation added as
  `digests/11.digest.md` (`59f0ddb`, `f88b216`); issue logs refreshed (`752ac64`).

## 2026-07-21
- **#10** — K4TEL added two comments steering the issue to implementation: use PDF-to-MD / DOCX-to-MD
  tools and record **page borders + as many visual layout cues as possible as HTML comments inside the
  Markdown**, and made "the list of all possible visual layout pieces" — an **exhaustive taxonomy** of
  cues with their exact MD/HTML encodings (`<!-- PAGE_BREAK -->`, `<!-- BBOX -->`, `<!-- FONT -->`,
  `<!-- HEADER_START -->`, `~~strike~~`, footnotes, tables, `<!-- WATERMARK -->`, …).

## 2026-07-22
- **#10** — **First implementation landed** (`3e7a909`): a visually-rich Markdown converter for **DOCX
  + digital-born PDF** (scope confirmed with K4TEL; standalone pre-convert CLI; pragmatic-core cue
  coverage). New `api_util/` modules — `layout_md.py` (dependency-free cue vocabulary + `CUE_SCHEMA`,
  single-sourcing the taxonomy), `docx_to_md.py` (python-docx), `pdf_to_md.py` (pdfplumber, with the
  decode-sanity check → `NEEDS_OCR` for text-less/garbled pages), `doc_to_visual_md.py` (dispatcher +
  CLI). Lands a `.md` in `INPUT_DIR`, consumed unchanged by `run_document_level()` — HTML-comment cues
  pass through as inert text. Optional `requirements_docmd.txt` (python-docx + pdfplumber, both MIT);
  four hermetic test modules. README + `plans/10.plan.md` (§6) + `digests/10.digest.md` refreshed.
  Full suite **255 passed, 3 skipped**; ruff clean. The scanned/curve-only **OCR path stays deferred**
  (pages flagged `NEEDS_OCR`; tool choice benchmark-gated under hub #22).

---
_Timeline index rebuilt for `atrium-llm-enrich` on 2026-07-22 against `test` HEAD `3e7a909`, the
per-issue digests/plans, and the dated commit history — replacing an earlier copy that had been left
as `atrium-translator`'s DEVLOG (flagged in `digests/10.digest.md`). This file is a derived reading
aid in `agent_dev_logs/`; nothing is removed from the issues themselves._
