---
name: atrium-llm-enrich
description: Extracts vocabulary-guided archaeological keywords from digitized document lines using LLM backends (OpenRouter remote API or local Ollama) - each line gets Czech + English key terms, a TEATER/AMCR thematic category constrained to the controlled vocabulary, and a confidence score. Use this skill for semantic enrichment of archival text after OCR and quality filtering, when NLP-statistical keywords are not enough and vocabulary-grounded semantic categories are needed.
---

# ATRIUM LLM Enrichment Skill 🗝️

This skill provides agent access to the **ATRIUM LLM Enrichment** service -
vocabulary-constrained keyword extraction over an LLM. It follows a
**server-client** design: a FastAPI server (in `service/`) drives the LLM
backends and the TEATER/AMCR vocabulary prompting, and a zero-dependency
client script (`scripts/atrium_keywords.py`) is the only thing the agent
calls directly.

## Operational Requirements ⚙️

- **Server**: a running instance is required. Default `http://localhost:8000`;
  override with `--base-url` or the `ATRIUM_LE_URL` environment variable.
- **Client dependencies**: none - `scripts/atrium_keywords.py` uses only the
  Python 3 standard library.
- **Server dependencies**: Docker (recommended, compose `api` profile) or a
  Python venv with `requirements_remote.txt` + `service/requirements.txt`
  (torch-free). A backend must be configured: `OPENROUTER_API_KEY` (+
  `OPENROUTER_MODEL` in `llm_config.txt`) for `openrouter`, or a reachable
  Ollama server (`OLLAMA_HOST`, `OLLAMA_MODEL`) for `ollama`.
- **First launch**: the TEATER/AMCR vocabulary auto-syncs from the AMCR
  OAI-PMH API when its cache file is missing - minutes, network-bound. Do
  **not** treat a slow first start as failure; `/health?deep=true` reports
  readiness.
- **Limits**: 5 MB per upload, 300 lines per request, 1 concurrent extraction
  (HTTP 429 when busy). **This is the slowest ATRIUM service** - a full
  document takes minutes.

## Backends & vocabulary 🗝️

| Backend      | Where it runs                    | Requirements                               |
|--------------|----------------------------------|--------------------------------------------|
| `openrouter` | remote LLM-as-a-service          | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`   |
| `ollama`     | local Ollama server              | `OLLAMA_HOST` reachable, `OLLAMA_MODEL`    |
| `local`      | in-process transformers/vLLM     | **CLI-only on the development branch**; API answers 501 |

Per line the LLM must pick the single most relevant **TEATER/AMCR category**
from the controlled vocabulary (it cannot invent categories), extract Czech
key terms found in the line itself, translate them to English, and report a
calibrated confidence score. Meta-text lines are categorized
`Nerelevantní (meta-text)` with no keywords.

## Workflows 🪄

### 1. Ensure the server is running

```bash
bash scripts/server.sh          # Docker Compose api profile (or local fallback)
bash scripts/server.sh --local  # force local uvicorn (no Docker)
```

dempotent: exits immediately if GET /info already answers; waits for
first-run vocabulary sync.

### 2. Extract keywords

```bash
# Plain text lines (one per line)
python3 scripts/atrium_keywords.py small_data_samples/lines_sample.txt

# CSV with text[,page_num,line_num] columns, explicit backend, fewer keywords
python3 scripts/atrium_keywords.py small_data_samples/lines_sample.csv --backend ollama --top-k 5

# TEITOK XML page, full JSON envelope
python3 scripts/atrium_keywords.py page.teitok.xml --format json

# Inline lines from stdin (no file needed)
printf 'V sondě S3 byly odkryty základy kostela.\n' | python3 scripts/atrium_keywords.py - --doc-id CTX1

# Discover capabilities, configured backends, and limits
python3 scripts/atrium_keywords.py --info
```

### 3. Interpret output

- table (default): DOC, PAGE, LINE, CATEGORY, CONF, KEYWORDS_CS rows plus
a one-line run summary (backend, model, processed/filtered/error counts) on
stderr.
- csv: adds KEYWORDS_EN, complete keyword lists for downstream tabular use.
- json: the full envelope - doc_id, backend, model, stats, and per
line keywords_cs, keywords_en, category, confidence, text.

## Agent Guidelines 🤖

1. Backend selection: omit --backend to use the server default. Choose
ollama only when a local Ollama server is running; local is CLI-only on the
development branch (the API answers 501 - use another backend here).
2. Confidence discipline: treat confidence < 0.7 as tentative - surface
the category with its score rather than asserting it; downstream filters
commonly threshold on this field.
3. Prefer --format json (or csv) when the result feeds further
processing; the table truncates keyword lists for readability.
4. For full request/response schemas, fetch GET /openapi.json from the
running server (Swagger UI at /docs).
5. Exit code 2 (unreachable): start the server (bash scripts/server.sh)
and retry once. Exit code 3 (server error): the client already retried
502/503/504 three times - check GET /health?deep=true (vocabulary +
backend readiness) and server logs; do not loop. HTTP 429 means the single
concurrent extraction slot is taken - wait and retry, do not hammer.
6. Budget limits: 300 lines per request and minutes of runtime - split
large documents, run the quality filter (atrium-alto-postprocess) first so
only meaningful lines reach the LLM, and tell the user what you did.
7. Do not bypass the API by importing the LLM client code directly - the
server is the supported entry point and enforces the vocabulary contract
and rate limits.

## Acknowledgements & Citations 🙏

The models and dataset are developed within the [ATRIUM](https://atrium-research.eu/)
project at ÚFAL, Charles University, with data hosted on
[LINDAT/CLARIAH-CZ](https://lindat.cz). If you use this service for research, cite the
repository's `CITATION.cff` and the LINDAT dataset record
(http://hdl.handle.net/20.500.12800/1-6184).
