# Strategy — New repo `ufal/atrium-llm-enrich` (Issue #24: LLM applications to data)

## Context
Issue [#24](https://github.com/ufal/atrium-project/issues/24) ("LLM applications to data",
Q4 · WP3, labels `development` / `beginning` / `a big one`) collects LLM-driven data-processing
subtasks — run models **locally** (multi-GPU, per hub #26/#27) and **remotely as a service**
(OpenRouter, per the #24 comment thread). In [#24 comment 4831318808](https://github.com/ufal/atrium-project/issues/24#issuecomment-4831318808)
K4TEL proposed spinning these out into a dedicated repo.

The LLM machinery today lives inside `ufal/atrium-nlp-enrich`, entangled with the NameTag 3 +
UDPipe 2 NER/morphosyntax API pipeline it does **not** need. The goal is a focused, LLM-only repo
that (a) copies the reusable LLM engine, (b) drops NameTag/UDPipe, (c) keeps CSV + plain-text + XML
input paths, and (d) adds remote (OpenRouter) and lightweight-local (Ollama) client backends —
with multi-GPU inference dependencies present from day one.

**Two decisions confirmed with the user:**
- **Name:** `atrium-llm-enrich` — parallels `atrium-nlp-enrich` as its LLM-based sibling
  (`enrich` = the task, add a semantic layer; `llm` signals the local+remote remit).
- **Relationship:** **copy / duplicate.** `atrium-nlp-enrich` is left **untouched** (keeps its LLM
  keyword path #6). No cross-repo refactor; some duplication to reconcile later via a shared package.

> **Scope note (important).** This session's GitHub access is limited to `ufal/atrium-project` and
> `ufal/atrium-nlp-enrich`. The new repo `ufal/atrium-llm-enrich` **cannot be created or pushed
> from here** — that is an owner action (`gh repo create`) or needs the session scope extended.
> What *is* executable in-scope is recording this strategy where the team already tracks work
> (see "In-scope deliverable" below); the physical bootstrap is the owner step in "Repo bootstrap".

---

## What to COPY from `atrium-nlp-enrich` (LLM core — reusable, engine untouched)
| Source file                                                                                                                        | Role in new repo                                                                                                                                                                                                                             |
|------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `llm_run.py`                                                                                                                       | Standalone entry point. Already reads **CSV** and **`*.teitok.xml`**, builds context windows, validates Pydantic-schema JSON.                                                                                                                |
| `llm_utils.py`                                                                                                                     | Engine: `MODEL_REGISTRY`, `transformers`+BnB and **vLLM** backends, guided decoding, `read_input_rows()`, `get_context_window()`, `process_document(_vllm)`, `load_vllm_engine()` (already wires `tensor_parallel_size` + `cpu_offload_gb`). |
| `vocab_manager.py`                                                                                                                 | TEATER thesaurus vocabulary manager (`requests`-only, standalone).                                                                                                                                                                           |
| `atrium_paradata.py`, `para_licenses.py`, `para_config.txt`                                                                        | Shared provenance logger (ecosystem convention; also in `atrium-project/docs/templates/shared/`).                                                                                                                                            |
| `api_util/teitok_read.py`                                                                                                          | Dependency-free TEITOK reader (imported by `llm_utils`/`llm_run`).                                                                                                                                                                           |
| `api_util/teitok_alto.py`                                                                                                          | ALTO↔TEITOK converter incl. PNG/JPEG/TIFF header readers — **base for the XML→Markdown work**.                                                                                                                                               |
| `api_util/flexiconv_convert.py` + `requirements_flexiconv.txt`                                                                     | txt/pdf/docx/html/md → TEITOK adapter (the "just texts from txt" path, nlp #10).                                                                                                                                                             |
| `llm_config.txt`, `requirements_llm.txt`, `requirements.txt`                                                                       | Config + deps. `requirements_llm.txt` already pins the **multi-GPU stack** (`vllm>=0.9.0`, `torch==2.7.0`, `transformers`, `bitsandbytes`).                                                                                                  |
| `LICENSE`, `.gitignore`, `.pre-commit-config.yaml`, `ruff.toml`, `.coveragerc`, `pytest.ini`, `Dockerfile`, `docker-compose*.yaml` | Repo hygiene, adapted.                                                                                                                                                                                                                       |

## What to DROP (NameTag + UDPipe application)
- Shell pipeline: `api_1_manifest.sh`, `api_2_udp.sh`, `api_3_nt.sh`, `api_4_stats.sh`,
  `api_flexiconv.sh`, `setup_api_service.sh`, `config_api.txt`.
- `api_util/`: `call_nametag.py`, `call_udpipe.py`, `summarize_nt_udp.py`, `build_manifest_row.py`,
  `bbox_scale.py`, `api_common.sh`, `test_cli_orchestration.py`.
- `run_pipeline.py` (merged NER+UDP+KW+LLM pipeline), `fix_teitok_bboxes.py`.
- `service/` (FastAPI `/enrich` wraps the NER pipeline; `service/api.py` imports nametag/udpipe) —
  drop now; an **LLM-only** service can follow later.
- Statistical keyword extractor `keywords.py` + `kw_config.txt` + `ker_data/` — **omit initially**
  (not an LLM application; YAKE/KeyBERT/KER can be re-added later as a baseline if wanted).
- NER/UDP tests: `tests/test_nametag.py`, `tests/tets_udpipe.py`, `tests/test_conllu_processing.py`,
  `tests/test_remote_apis.py`.

## What to ADD (new work — the point of #24)
1. **`openrouter_client.py`** — remote LLM-as-a-service. Per #24 comment: curl + secret key already
   works; implement Python routines. Env `OPENROUTER_API_KEY`. Explore **file-attachment** options
   and **provider routing for a local-only / no-logging data source** (data-sovereignty). **Reuse
   the existing Pydantic schemas + `validate_llm_output()`** so remote and local share one output
   contract.
2. **`ollama_client.py`** — local Ollama server (`/api/chat`, `/api/generate`) with the JSON `format`
   schema for structured output + model-pull handling. The lightweight local alternative to the
   heavy transformers/vLLM path.
3. **`api_util/xml_to_md.py`** — the **TEITOK/ALTO → Markdown** converter. Render whole XML docs to
   Markdown/plain text so entire documents can be fed to LLMs (local *or* OpenRouter attachment) —
   document-level input, complementing the existing line-level CSV/TEITOK row reader. Build on
   `teitok_alto.py` + `teitok_read.py`.
4. **Multi-GPU from day one.** Keep `requirements_llm.txt` pins and the `llm_config.txt` knobs
   `BACKEND=vllm`, `TENSOR_PARALLEL_SIZE`, `CPU_OFFLOAD_GB`, `GPU_MEMORY_UTILIZATION` (the #26 UVA
   offload + #27 8×H100 tensor-parallel recipes are already documented there and wired in
   `load_vllm_engine()`). Ship `docker-compose.gpu.yaml` with GPU reservations.
5. **Thin backend dispatch.** Add a `BACKEND ∈ {transformers, vllm, ollama, openrouter}` selector so
   the shared front-end (`read_input_rows` → `get_context_window` → schema validation) routes to
   local or remote without duplicating prompt-building/validation. `requirements_remote.txt` holds
   the light client deps (`httpx`/`openrouter`/`ollama`) so remote-only users skip the GPU stack.

## Proposed repo layout
```
atrium-llm-enrich/
├── README.md  LICENSE(MIT)  CITATION.cff(v0.1.0)  CONTRIBUTING.md
├── .gitignore .pre-commit-config.yaml ruff.toml .coveragerc pytest.ini
├── llm_config.txt                 # registry knobs incl. multi-GPU (TP / cpu_offload)
├── requirements.txt  requirements_llm.txt(GPU)  requirements_remote.txt(NEW)  requirements_flexiconv.txt
├── llm_run.py  llm_utils.py  vocab_manager.py            # copied engine
├── atrium_paradata.py  para_licenses.py  para_config.txt # provenance
├── openrouter_client.py  ollama_client.py                # NEW backends
├── api_util/{teitok_read.py, teitok_alto.py, flexiconv_convert.py, xml_to_md.py(NEW)}
├── docker-compose.yaml  docker-compose.gpu.yaml  Dockerfile
├── tests/                          # LLM-only (no nametag/udpipe)
└── .github/workflows/              # thin callers → ufal/atrium-project *.reusable.yml@test
```

## Reuse existing ecosystem conventions (do not reinvent)
- **CI/CD:** copy the caller examples in `atrium-project/docs/templates/workflows/`
  (`docker`, `gpu-inference`, `pre-commit`, `codeql`, `secret-scan`, `scheduled-smoke`,
  `dependabot`) — each is a thin caller into `ufal/atrium-project/.github/workflows/*.reusable.yml@test`.
  A `gpu-inference` reusable workflow already exists — reuse it for this GPU-heavy repo.
- **Shared files:** `atrium_paradata.py`, `para_licenses.py`, `ruff.toml`, `.coveragerc`,
  `CONTRIBUTING.md` come from `atrium-project/docs/templates/` — keep them in sync, don't fork logic.
- **Paradata:** `atrium_paradata.py` already reads `ATRIUM_RUNNER_IMAGE/REPO/REF`; publish a
  self-identifying GHCR image (hub #18 pattern) so runs are provenance-tagged.

---

## In-scope deliverable (what I execute on approval)
Because the new repo is out of session scope, record the strategy the way every other issue is
tracked, on branch `claude/llm-data-processing-repo-6on1sl` in **`ufal/atrium-project`**:
1. `agent_dev_logs/plans/24.plan.md` — this strategy (matches the existing `NN.plan.md` format).
2. `agent_dev_logs/digests/24.digest.md` — short digest (matches existing `NN.digest.md`).
3. Append a `#24` entry to `agent_dev_logs/DEVLOG.md`.
4. Optionally post a concise summary comment on issue #24 (confirm before posting — outward-facing).

No changes to `atrium-nlp-enrich` (copy strategy ⇒ it stays untouched).

## Repo bootstrap (owner action / when scope allows)
```bash
gh repo create ufal/atrium-llm-enrich --private --description "LLM-based semantic enrichment (local multi-GPU + OpenRouter/Ollama) for ATRIUM archival text"
# seed from the COPY manifest above (git-copy the listed files from atrium-nlp-enrich),
# drop the NameTag/UDPipe set, add the 3 NEW modules, wire .github callers @test,
# update CITATION.cff (title/version 0.1.0) and README, then:
git add -A && git commit -m "Initial atrium-llm-enrich: LLM engine (local+remote) from nlp-enrich, minus NameTag/UDPipe"
git push -u origin main
# open a mirror issue in the new repo; cross-link ufal/atrium-project#24
```

## Verification
- **Local (transformers):** `python llm_run.py` with a small model (e.g. `qwen3-8b`) over a sample
  `data_samples/*.csv` and a `*.teitok.xml` → confirm Pydantic-schema JSON + paradata written.
- **Multi-GPU (vLLM):** run with `BACKEND=vllm`, `TENSOR_PARALLEL_SIZE=N` (+ optional
  `CPU_OFFLOAD_GB` from the #26/#27 recipe) → capture startup inference-param summary +
  `log_gpu_memory()` + tok/s.
- **OpenRouter:** `OPENROUTER_API_KEY=… python openrouter_client.py --input sample.csv --model <cheap>`
  → same schema validates; exercise the file-attachment path with a Markdown-rendered doc.
- **Ollama:** `ollama serve` + `python ollama_client.py --input sample.md --model qwen2.5:7b`
  → structured JSON via `format` schema.
- **Converter:** `python api_util/xml_to_md.py sample.teitok.xml` → Markdown; feed to both clients.
- **CI:** `ruff check` + `pytest -m "not slow"` green via the pre-commit/codeql callers.