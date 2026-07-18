# LLM enrichment API service 🗝️

FastAPI entry point for the ATRIUM llm-enrich pipeline: upload document text
lines, get back **vocabulary-guided archaeological keywords** (Czech + English)
with a TEATER/AMCR thematic category and a confidence score per line. The
service wraps the existing remote/lightweight LLM clients
(`openrouter_client.py`, `ollama_client.py`) through `llm_client_shared.py`;
the torch stack is **not** imported (`backend=local` is CLI-only on the
development branch and answers HTTP 501 here). The service version is read from
`para_config.txt` `[tool]` (single source of truth, never hard-coded).

## Quick start

```bash
pip install -r requirements_remote.txt -r service/requirements.txt
export OPENROUTER_API_KEY=sk-or-...          # or run a local Ollama server
uvicorn service.api:app --host 0.0.0.0 --port 8000
# or:
docker compose --profile api up -d
```

A minimal demo frontend is served at `http://localhost:8000/frontend/`.

> [!NOTE]
> On first start the TEATER/AMCR vocabulary is auto-synced from the AMCR
> OAI-PMH API when the cache file is missing — this can take minutes and needs
> outbound network access. `/health?deep=true` reports readiness.

## Endpoints

| Method | Path                     | Purpose                                                                                       |
|--------|--------------------------|-----------------------------------------------------------------------------------------------|
| GET    | `/`                      | redirects to the demo frontend                                                                |
| GET    | `/info`                  | service identity + capabilities: `service`, `version`, `endpoints`, `limits`, backends, vocab |
| GET    | `/health`                | liveness probe; `?deep=true` verifies vocabulary readiness + the selected backend (503 on fail)|
| POST   | `/extract_keywords`      | **file entry point** — upload TXT / CSV / TEITOK XML                                          |
| POST   | `/extract_keywords_text` | same pipeline for inline JSON lines                                                           |

### `POST /extract_keywords` (multipart form)

| Field     | Default          | Notes                                                                    |
|-----------|------------------|--------------------------------------------------------------------------|
| `file`    | *required*       | `.txt` (plain lines), `.csv` (needs a `text` column), or `.teitok.xml`   |
| `backend` | server default   | `openrouter` \| `ollama` (\| `local` → 501, CLI-only)                    |
| `top_k`   | `10`             | max keywords per line, 1–50                                              |

Convert ALTO XML to TEITOK/CSV first (`api_util/` tooling on the development
branch) — raw ALTO uploads fail with 422 (no TEITOK rows), other suffixes with 415.

### `POST /extract_keywords_text` (JSON)

```json
{ "doc_id": "CTX1", "lines": ["Výzkum odhalil základy kostela."],
  "backend": "openrouter", "top_k": 10 }
```

### Response envelope

```json
{
  "doc_id": "CTX1",
  "backend": "openrouter",
  "model": "openai/gpt-4o-mini",
  "vocabulary": "TEATER/AMCR",
  "stats": {"processed": 2, "skipped_filter": 0, "skipped_error": 0, "aborted": 0},
  "lines": [
    {
      "page": 1, "line": 1,
      "text": "Výzkum odhalil základy kostela.",
      "keywords_cs": ["základy kostela"],
      "keywords_en": ["church foundations"],
      "category": "kostel",
      "confidence": 0.92
    }
  ]
}
```

| Field         | Type  | Description                                                             |
|---------------|-------|-------------------------------------------------------------------------|
| `keywords_cs` | list  | Czech terms extracted from the line (not copied from the vocabulary)    |
| `keywords_en` | list  | English translations of `keywords_cs`                                   |
| `category`    | str   | the single most relevant TEATER/AMCR vocabulary category                |
| `confidence`  | float | category confidence [0–1] (usable as a filter threshold)                |
| `stats`       | obj   | per-request run statistics (`processed`/`skipped_filter`/`skipped_error`/`aborted`) |

## Errors

| Code        | Meaning                                                                    |
|-------------|-----------------------------------------------------------------------------|
| 413         | upload exceeds `MAX_UPLOAD_MB`, or line count exceeds `MAX_LINES`          |
| 415         | unsupported media type (e.g. raw ALTO XML)                                 |
| 422         | unusable input (no text rows, bad `backend`/`top_k`)                       |
| 429         | busy — `MAX_CONCURRENT_JOBS` reached; retry later                          |
| 501         | `backend=local` requested (CLI-only in this release)                       |
| 503         | backend unconfigured/unreachable, or vocabulary warmup failed              |
| 504         | extraction exceeded `API_JOB_TIMEOUT`                                      |
| 502/503/504 | **clients retry 3×** (warmup/proxy)                                        |

## Configuration (environment)

| Variable              | Default      | Meaning                                                            |
|-----------------------|--------------|--------------------------------------------------------------------|
| `BACKEND`             | `openrouter` | default backend when the request does not specify one              |
| `OPENROUTER_API_KEY`  | —            | required for the openrouter backend                                |
| `OPENROUTER_MODEL`    | (config)     | falls back to `llm_config.txt`                                     |
| `OLLAMA_HOST`         | `http://localhost:11434` | Ollama server for the ollama backend                   |
| `OLLAMA_MODEL`        | (config)     | falls back to `llm_config.txt`                                     |
| `MAX_UPLOAD_MB`       | `5`          | upload size guard                                                  |
| `MAX_LINES`           | `300`        | per-request line cap (LLM cost/time budget guard)                  |
| `MAX_CONCURRENT_JOBS` | `1`          | concurrent extractions (LLM calls are the slowest in the family)   |
| `API_JOB_TIMEOUT`     | `1800`       | per-request timeout in seconds                                     |
| `ALLOWED_ORIGINS`     | `*`          | CSV of CORS origins                                                |
| `HF_TOKEN`            | —            | only the batch `llm` image needs it (not the API)                  |

## How it works

Requests are normalized to the canonical `text[,page_num,line_num,categ,quality_score]`
row form, then run through `llm_client_shared.run_line_level()` — the same
quality filter, ±2-line context windows, vocabulary-constrained JSON schema,
and lenient validation used by the batch clients — so API results are
comparable with CLI runs. Line-level records are mapped to the
`keywords_cs/keywords_en/category/confidence` envelope above. LLM calls are
the slowest in the ATRIUM family: the API is synchronous with a strict
concurrency guard; adopting nlp-enrich's async jobs pattern is the planned
fast-follow if sync proves impractical.

## Frontend

`service/frontend/` is a minimal standalone HTML/JS client mounted at
`/frontend`: paste lines or upload a file, pick a backend, and inspect the
keyword table. It links the live API docs (`/docs`, `/openapi.json`).

## Tests

The hermetic API test suite (mocked chat function, no network) lives on the
development ([`test`](https://github.com/ufal/atrium-llm-enrich/tree/test))
branch:

```bash
pytest tests/test_service_api.py
```
