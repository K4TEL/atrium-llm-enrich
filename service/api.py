"""
service/api.py — FastAPI surface for the llm-enrich keyword extraction.

Mirrors the atrium-nlp-enrich service layout: a thin HTTP wrapper over the
existing remote/local-lightweight LLM clients, dispatched through
llm_client_shared.py. The torch stack is deliberately NOT imported here —
`backend=local` (transformers/vLLM via llm_run.py) stays CLI-only for now and
the API answers 501 for it, keeping this service installable from
requirements_remote.txt alone (the repo's established constraint).
"""

from __future__ import annotations

import asyncio
import configparser
import csv
import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# service/ lives one level below the repo root where all pipeline modules sit.
_SERVICE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SERVICE_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import ollama_client  # noqa: E402
import openrouter_client  # noqa: E402
import requests  # noqa: E402
from llm_client_shared import (  # noqa: E402
    build_schema,
    build_system_prompt,
    load_config,
    run_line_level,
)
from vocab_manager import VocabularyManager  # noqa: E402

# ── operator-tunable limits ───────────────────────────────────────────────────
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "5"))
MAX_LINES = int(os.environ.get("MAX_LINES", "300"))
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
API_JOB_TIMEOUT = int(os.environ.get("API_JOB_TIMEOUT", "1800"))
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "32000"))
CONTEXT_RESERVED = 4000  # kept for the model's answer + user message
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

SERVICE_NAME = "atrium-llm-enrich"
API_ENDPOINTS = ["/info", "/health", "/extract_keywords", "/extract_keywords_text"]
_ALLOWED_BACKENDS = ("openrouter", "ollama", "local")
_INPUT_SUFFIXES = (".txt", ".csv", ".xml")

_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_config: Dict[str, str] = {}
_prompt_state: Dict[str, Any] = {}  # system_prompt, EnrichmentModel, schema_json, vocab_path
_chat_state: Dict[str, Any] = {}  # per-backend cached chat_fn + model name


def _read_tool_version() -> str:
    """Read the tool version from para_config.txt [tool] section (single source
    of truth — validated against CITATION.cff by security.reusable.yml)."""
    config = configparser.ConfigParser()
    config.read(_REPO_ROOT / "para_config.txt", encoding="utf-8")
    version = config.get("tool", "version", fallback="unknown")
    return version[1:] if version.lower().startswith("v") else version


def _default_backend() -> str:
    return os.environ.get("BACKEND") or _config.get("BACKEND") or "openrouter"


def _openrouter_key() -> Optional[str]:
    return os.environ.get("OPENROUTER_API_KEY") or _config.get("OPENROUTER_API_KEY")


def _ollama_host() -> str:
    return (
        os.environ.get("OLLAMA_HOST") or _config.get("OLLAMA_HOST") or ollama_client.DEFAULT_OLLAMA_HOST
    ).rstrip("/")


def _prepare_prompts() -> None:
    """Load the TEATER/AMCR vocabulary and build the line-level prompt/schema
    once; auto-syncs the vocabulary from the AMCR API when the cache file is
    missing (can take minutes on a cold start)."""
    vocab_path = _config.get("VOCAB_PATH", "data_samples/teater_nested_vocab.json")
    if not Path(vocab_path).is_absolute():
        vocab_path = str(_REPO_ROOT / vocab_path)
    vocab_data = VocabularyManager(vocab_path=vocab_path).load()
    system_prompt, terms = build_system_prompt(vocab_data, max_tokens=CONTEXT_WINDOW - CONTEXT_RESERVED)
    model_cls = build_schema(terms)
    _prompt_state.update(
        {
            "system_prompt": system_prompt,
            "EnrichmentModel": model_cls,
            "schema_json": model_cls.model_json_schema(),
            "vocab_path": vocab_path,
            "num_terms": len(terms),
        }
    )


def _get_chat_fn(backend: str):
    """Build (and cache) a llm_client_shared.ChatFn for the chosen backend.
    Raises HTTPException with an actionable detail when unconfigured."""
    if backend in _chat_state:
        return _chat_state[backend]["chat_fn"], _chat_state[backend]["model"]

    if backend == "openrouter":
        api_key = _openrouter_key()
        if not api_key:
            raise HTTPException(503, "OpenRouter backend not configured: set OPENROUTER_API_KEY.")
        model = os.environ.get("OPENROUTER_MODEL") or _config.get("OPENROUTER_MODEL")
        if not model:
            raise HTTPException(503, "OpenRouter backend not configured: set OPENROUTER_MODEL in llm_config.txt.")
        session = requests.Session()
        headers = openrouter_client._build_headers(api_key, None, None)
        chat_fn = openrouter_client.make_chat_fn(
            session, headers, model, _prompt_state["schema_json"], max_retries=3, timeout=300, provider_block=None
        )
    elif backend == "ollama":
        model = os.environ.get("OLLAMA_MODEL") or _config.get("OLLAMA_MODEL")
        if not model:
            raise HTTPException(503, "Ollama backend not configured: set OLLAMA_MODEL in llm_config.txt.")
        host = _ollama_host()
        session = requests.Session()
        try:
            ollama_client.ensure_model_pulled(host, model, session, timeout=300)
        except Exception as exc:
            raise HTTPException(503, f"Ollama unreachable at {host}: {exc}") from exc
        chat_fn = ollama_client.make_chat_fn(session, host, model, _prompt_state["schema_json"], 3, 300)
    elif backend == "local":
        raise HTTPException(
            501,
            "backend=local (transformers/vLLM) is CLI-only in this release — run llm_run.py, "
            "or use backend=openrouter / backend=ollama via the API.",
        )
    else:
        raise HTTPException(422, f"backend must be one of {_ALLOWED_BACKENDS}")

    _chat_state[backend] = {"chat_fn": chat_fn, "model": model}
    return chat_fn, model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _config.update(load_config(str(_REPO_ROOT / "llm_config.txt")))
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, _prepare_prompts)
    except Exception as exc:  # degraded start — surfaced via /health
        print(f"[WARN] Vocabulary/prompt warmup failed: {exc}")
    yield


app = FastAPI(
    title="ATRIUM llm-enrich API",
    version=_read_tool_version(),
    description="Vocabulary-guided LLM keyword extraction for archaeological archival text.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if (_SERVICE_DIR / "frontend").exists():
    app.mount("/frontend", StaticFiles(directory=str(_SERVICE_DIR / "frontend"), html=True), name="frontend")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/frontend")


@app.get("/info")
async def info() -> Dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": app.version,
        "endpoints": API_ENDPOINTS,
        "limits": {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_lines": MAX_LINES,
            "max_concurrent_jobs": MAX_CONCURRENT_JOBS,
            "api_job_timeout_s": API_JOB_TIMEOUT,
        },
        "backends": {
            "default": _default_backend(),
            "openrouter": {
                "configured": bool(_openrouter_key()),
                "model": os.environ.get("OPENROUTER_MODEL") or _config.get("OPENROUTER_MODEL"),
            },
            "ollama": {
                "host": _ollama_host(),
                "model": os.environ.get("OLLAMA_MODEL") or _config.get("OLLAMA_MODEL"),
            },
            "local": "cli-only (llm_run.py; API answers 501)",
        },
        "vocabulary": {
            "source": "TEATER/AMCR",
            "path": _prompt_state.get("vocab_path"),
            "num_terms": _prompt_state.get("num_terms"),
            "ready": bool(_prompt_state),
        },
        "input_formats": ["TXT (plain lines)", "CSV (text[,page_num,line_num,categ,quality_score])", "TEITOK XML"],
    }


@app.get("/health")
async def health(deep: bool = False) -> JSONResponse:
    """Liveness (shallow) / readiness (deep=true, vocabulary + backend) probe."""
    if not deep:
        return JSONResponse({"status": "ok"})

    if not _prompt_state:
        return JSONResponse(
            {"status": "degraded", "detail": "vocabulary/prompt warmup failed or still running"},
            status_code=503,
        )

    backend = _default_backend()
    detail: Dict[str, Any] = {"status": "ok", "backend": backend, "vocabulary_ready": True}
    if backend == "openrouter":
        if not _openrouter_key():
            return JSONResponse(
                {"status": "degraded", "detail": "OPENROUTER_API_KEY not set", "backend": backend},
                status_code=503,
            )
        detail["api_key_present"] = True
    elif backend == "ollama":
        host = _ollama_host()
        try:
            requests.get(f"{host}/api/tags", timeout=5).raise_for_status()
            detail["ollama_reachable"] = True
        except Exception as exc:
            return JSONResponse(
                {"status": "degraded", "detail": f"Ollama unreachable at {host}: {exc}", "backend": backend},
                status_code=503,
            )
    elif backend == "local":
        detail["note"] = "backend=local is CLI-only; API requests answer 501"
    return JSONResponse(detail)


# ── request handling ───────────────────────────────────────────────────────────


def _rows_to_temp_csv(rows: List[Dict[str, Any]], tmpdir: str, doc_id: str) -> Path:
    """Materialize normalized rows as the canonical CSV the shared runner reads."""
    path = Path(tmpdir) / f"{doc_id}.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "page_num", "line_num", "categ", "quality_score"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "text": row.get("text", ""),
                    "page_num": row.get("page_num", 1),
                    "line_num": row.get("line_num", ""),
                    "categ": row.get("categ", ""),
                    "quality_score": row.get("quality_score", 1.0),
                }
            )
    return path


def _normalize_upload(filename: str, data: bytes, tmpdir: str) -> Path:
    """Turn an upload into a file path the shared runner understands."""
    name = (filename or "upload.csv").lower()
    doc_id = Path(name).stem.replace(".teitok", "") or "document"
    if name.endswith(".txt"):
        text = data.decode("utf-8-sig", errors="replace")
        rows = [
            {"text": line.strip(), "page_num": 1, "line_num": i}
            for i, line in enumerate(text.splitlines(), start=1)
            if line.strip()
        ]
        if not rows:
            raise HTTPException(422, "No usable text lines found in the TXT upload.")
        return _rows_to_temp_csv(rows, tmpdir, doc_id)
    if name.endswith(".teitok.xml") or name.endswith(".xml"):
        # read_input_rows special-cases *.teitok.xml — keep that suffix.
        path = Path(tmpdir) / f"{doc_id}.teitok.xml"
        path.write_bytes(data)
        return path
    if name.endswith(".csv"):
        path = Path(tmpdir) / f"{doc_id}.csv"
        path.write_bytes(data)
        return path
    raise HTTPException(
        415,
        "Unsupported media type. Allowed: .txt (plain lines), .csv (text column), .teitok.xml. "
        "Convert ALTO XML to TEITOK/CSV first (see api_util/).",
    )


def _run_extraction_sync(input_path: Path, backend: str, top_k: int) -> Dict[str, Any]:
    chat_fn, model = _get_chat_fn(backend)
    enriched, stats = run_line_level(
        input_path,
        chat_fn,
        _prompt_state["system_prompt"],
        _prompt_state["EnrichmentModel"],
    )
    lines = []
    for record in enriched:
        enrichment = record.get("enrichment") or {}
        lines.append(
            {
                "page": record.get("page"),
                "line": record.get("line"),
                "text": record.get("original_text"),
                "keywords_cs": (enrichment.get("extracted_keywords_cs") or [])[:top_k],
                "keywords_en": (enrichment.get("extracted_keywords_en") or [])[:top_k],
                "category": enrichment.get("teater_category", ""),
                "confidence": enrichment.get("confidence_score"),
            }
        )
    return {
        "doc_id": input_path.stem.replace(".teitok", ""),
        "backend": backend,
        "model": model,
        "vocabulary": "TEATER/AMCR",
        "stats": stats,
        "lines": lines,
    }


async def _extract_common(input_path: Path, backend: str, top_k: int):
    if backend not in _ALLOWED_BACKENDS:
        raise HTTPException(422, f"backend must be one of {_ALLOWED_BACKENDS}")
    if not (1 <= top_k <= 50):
        raise HTTPException(422, "top_k must be between 1 and 50")
    if not _prompt_state:
        raise HTTPException(503, "Vocabulary/prompt warmup failed or still running; check /health?deep=true.")

    from llm_client_shared import read_input_rows

    try:
        num_rows = len(read_input_rows(input_path))
    except Exception as exc:
        raise HTTPException(422, f"Could not parse input: {exc}") from exc
    if num_rows == 0:
        raise HTTPException(422, "No usable rows found in input.")
    if num_rows > MAX_LINES:
        raise HTTPException(413, f"Input too large: {num_rows} lines > {MAX_LINES} (LLM budget guard).")

    if _semaphore.locked():
        raise HTTPException(429, "Server busy; max concurrent extractions reached. Retry later.")

    async with _semaphore:
        loop = asyncio.get_event_loop()
        try:
            envelope = await asyncio.wait_for(
                loop.run_in_executor(None, _run_extraction_sync, input_path, backend, top_k),
                timeout=API_JOB_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(504, "Keyword extraction timed out.") from exc
    return JSONResponse(envelope)


@app.post("/extract_keywords")
async def extract_keywords(
    file: UploadFile = File(...),  # noqa: B008
    backend: str = Form(""),
    top_k: int = Form(10),
):
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_MB} MB.")
    resolved_backend = backend or _default_backend()
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = _normalize_upload(file.filename, data, tmpdir)
        return await _extract_common(input_path, resolved_backend, top_k)


@app.post("/extract_keywords_text")
async def extract_keywords_text(payload: Dict[str, Any]):
    lines = payload.get("lines")
    if not isinstance(lines, list) or not lines:
        raise HTTPException(422, "'lines' must be a non-empty list.")
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(lines, start=1):
        if isinstance(item, str) and item.strip():
            rows.append({"text": item.strip(), "page_num": 1, "line_num": i})
        elif isinstance(item, dict) and item.get("text"):
            rows.append(item)
    if not rows:
        raise HTTPException(422, "No usable text lines in 'lines'.")
    doc_id = str(payload.get("doc_id", "document"))
    backend = str(payload.get("backend") or _default_backend())
    top_k = int(payload.get("top_k", 10))
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = _rows_to_temp_csv(rows, tmpdir, doc_id)
        return await _extract_common(input_path, backend, top_k)
