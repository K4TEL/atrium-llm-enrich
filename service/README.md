# llm-enrich API service 🧠

LLM-based archaeological keyword extraction: text lines / documents in → per-line (or
per-document) `extracted_keywords_cs` / `extracted_keywords_en` out. The service version is
read from `para_config.txt` `[tool]` (single source of truth, never hard-coded).

It wraps the **torch-free** remote / lightweight-local engine (`llm_client_shared` +
`openrouter_client` / `ollama_client`), so it never needs the GPU stack.

## Quick start

```bash
pip install -r service/requirements.txt

# choose a backend and give it a key/model, then launch:
export LLM_BACKEND=openrouter OPENROUTER_API_KEY=sk-... OPENROUTER_MODEL=openai/gpt-4o-mini
uvicorn service.api:app --host 0.0.0.0 --port 8000
# or:
docker compose --profile api up -d
```

Without a configured backend the service still starts: `/info` and `/health` respond and
report `ready: false`, while the extraction endpoints return `503` until configured.

## Endpoints

| Method | Path                     | Purpose                                                                            |
|--------|--------------------------|------------------------------------------------------------------------------------|
| GET    | `/info`                  | service identity + capabilities: `service`, `version`, `endpoints`, `limits`, `backend`, `model`, `ready`, `supported_inputs`, `languages` |
| GET    | `/health`                | liveness probe; `?deep=true` additionally checks the backend is configured (503 on fail) |
| POST   | `/extract_keywords`      | extract keywords from an uploaded document                                          |
| POST   | `/extract_keywords_text` | extract keywords from an inline JSON `{"lines": [...]}` body                        |

### `POST /extract_keywords` (multipart form)

| Field  | Default    | Notes                                                                          |
|--------|------------|--------------------------------------------------------------------------------|
| `file` | *required* | `.csv` / `*.teitok.xml` → line-level; `.md` / `.txt` → document-level          |

```bash
curl -X POST "http://localhost:8000/extract_keywords" -F "file=@sample.csv"
curl -X POST "http://localhost:8000/extract_keywords_text" \
     -H "Content-Type: application/json" -d '{"lines": ["Výzkum odhalil základy gotického kostela."]}'
curl -s http://localhost:8000/info
```

### Response schema

```json
{
  "service": "atrium-llm-enrich",
  "doc_id": "sample",
  "backend": "openrouter",
  "model": "openai/gpt-4o-mini",
  "mode": "line",
  "results": [
    {
      "file_id": "sample",
      "page": "1",
      "line": "1",
      "original_text": "Výzkum odhalil základy gotického kostela.",
      "enrichment": {
        "extracted_keywords_cs": ["základy", "gotický kostel"],
        "extracted_keywords_en": ["foundations", "Gothic church"]
      }
    }
  ],
  "stats": {"processed": 1, "skipped_filter": 0, "skipped_error": 0}
}
```

| Field     | Type   | Description                                                       |
|-----------|--------|------------------------------------------------------------------|
| `service` | str    | canonical tool id (`atrium-llm-enrich`)                          |
| `doc_id`  | str    | document id derived from the upload filename                     |
| `backend` | str    | active LLM backend (`openrouter` / `ollama`)                    |
| `mode`    | str    | `line` (CSV/TEITOK) or `document` (MD/TXT)                       |
| `results` | list   | per-line/per-document records; `enrichment` holds the keywords   |
| `stats`   | object | processed / filtered / errored counts (+ `aborted` on abort)     |

## Errors

| Code        | Meaning                                                        |
|-------------|----------------------------------------------------------------|
| 413         | payload too large (`MAX_UPLOAD_MB`)                            |
| 422         | unusable input (missing filename, unsupported type, no lines)  |
| 500         | processing failure                                             |
| 502         | upstream LLM backend error (client retries)                    |
| 503         | backend not configured / not ready (client retries)            |

## Configuration (environment)

| Variable             | Default                    | Meaning                                        |
|----------------------|----------------------------|------------------------------------------------|
| `LLM_BACKEND`        | `openrouter`               | `openrouter` or `ollama`                       |
| `OPENROUTER_API_KEY` | —                          | key for the OpenRouter backend                 |
| `OPENROUTER_MODEL`   | —                          | OpenRouter model id                            |
| `OLLAMA_HOST`        | `http://localhost:11434`   | Ollama server URL                              |
| `OLLAMA_MODEL`       | —                          | Ollama model tag                               |
| `VOCAB_PATH`         | from `llm_config.txt`      | archaeological vocabulary JSON                  |
| `MAX_UPLOAD_MB`      | `10`                       | canonical upload limit                          |
| `ALLOWED_ORIGINS`    | `*`                        | CSV of CORS origins                             |
| `LLM_TIMEOUT`        | `300`                      | per-call read timeout (s)                       |

## How it works

On startup the service loads `llm_config.txt` + the archaeological vocabulary, builds the
system prompt and Pydantic schema once, and binds a `chat_fn` to the chosen backend. Each
request writes the upload to a temp file and calls `llm_client_shared.run_line_level` (CSV/
TEITOK) or `run_document_level` (MD/TXT) in a threadpool, so the event loop stays responsive.
Backend warmup failures are recorded rather than fatal, keeping `/info` and `/health` live.

## Tests

`tests/test_api_contract.py` (hermetic, `importorskip("fastapi")`) asserts the §4 meta-contract
against the in-process `app.openapi()`. Run: `pytest -m "not slow" tests/test_api_contract.py`.
