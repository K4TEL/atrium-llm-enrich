<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" title="Python Version"></a>
  <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/backend-transformers%20%7C%20vLLM-orange.svg" title="Local backends"></a>
  <a href="https://openrouter.ai/"><img src="https://img.shields.io/badge/remote-OpenRouter-6E56CF.svg" title="OpenRouter"></a>
  <a href="https://ollama.com/"><img src="https://img.shields.io/badge/local--light-Ollama-000000.svg" title="Ollama"></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/github/license/ufal/atrium-llm-enrich" title="MIT License"></a>
  <a href="https://atrium-research.eu/"><img src="https://img.shields.io/badge/funded%20by-ATRIUM-8A2BE2.svg" title="ATRIUM Project"></a>
</p>

---

# 🧠 ATRIUM LLM Enricher — Local (multi-GPU) & Remote (OpenRouter/Ollama) Semantic Enrichment

This project runs a Large Language Model over digitized archival text — either **locally**
(single-GPU `transformers` or multi-GPU `vLLM`) or **as a service** (`OpenRouter`, lightweight
local `Ollama`) — to map each text line or whole document onto the controlled TEATER/AMCR
archaeological vocabulary, extracting Czech/English keyword pairs and a thematic category with
a confidence score.

---

> [!NOTE]
> This repository is the **LLM-only sibling** of
> [`atrium-nlp-enrich`](https://github.com/ufal/atrium-nlp-enrich) [^2] — spun out per
> [ATRIUM issue #24](https://github.com/ufal/atrium-project/issues/24) "LLM applications to
> data" so the LLM engine isn't entangled with `atrium-nlp-enrich`'s NameTag 3 + UDPipe 2
> NER/morphosyntax pipeline. The LLM engine (`llm_utils.py`, `vocab_manager.py`,
> `atrium_paradata.py`, `api_util/`) is a **copy**, kept deliberately in sync rather than
> cross-repo-refactored — see [`llm_client_shared.py`](llm_client_shared.py) 📎 for how the
> remote/lightweight-local backends share logic with the local `transformers`/`vLLM` engine
> without importing torch.

## Table of contents

- [⚙️ Setup](#-setup)
- [Backends at a glance](#backends-at-a-glance)
- [Configuration (`llm_config.txt`)](#configuration-llm_configtxt)
- [Vocabulary Harvesting (`vocab_manager.py`)](#vocabulary-harvesting-vocab_managerpy)
- [Local Inference — `transformers` / `vLLM` (`llm_run.py`)](#local-inference--transformers--vllm-llm_runpy)
- [Remote Inference — OpenRouter (`openrouter_client.py`)](#remote-inference--openrouter-openrouter_clientpy)
- [Lightweight Local — Ollama (`ollama_client.py`)](#lightweight-local--ollama-ollama_clientpy)
- [Document-Level Input (`api_util/xml_to_md.py`)](#document-level-input-api_utilxml_to_mdpy)
- [🖥 Model Registry](#-model-registry)
- [📁 Inputs and Outputs](#-inputs-and-outputs)
- [📐 Document Understanding benchmark (`sample_stratify.py` + `bench_compare.py`)](#-document-understanding-benchmark-sample_stratifypy--bench_comparepy)
- [🐳 Docker](#-docker)
- [Paradata Logs](#paradata-logs)
- [Acknowledgements](#acknowledgements-)

## ⚙️ Setup

1. Create and activate a new virtual environment in the project directory 🖥.
2. Install the backend-agnostic base deps (needed by every entry point):
```bash
pip install -r requirements.txt
```
3. Install the deps for the backend(s) you intend to use:
```bash
# Local, transformers backend — single GPU, models ≤ 31 B (BnB 4-bit / AWQ / GGUF)
# Local, vLLM backend — multi-GPU, large models (≥ 70 B), Automatic Prefix Caching
pip install -r requirements_llm.txt

# Remote (OpenRouter) or lightweight-local (Ollama) — no torch/vllm/bitsandbytes
pip install -r requirements_remote.txt
```
*(Optional) For non-ALTO/non-TEITOK text input (txt/pdf/docx/html/md) via `flexiconv`:*
```bash
pip install -r requirements_flexiconv.txt
```
4. Review and update [`llm_config.txt`](llm_config.txt) 📎 — the only required change is
   `MODEL_KEY` (local backends) or `OPENROUTER_MODEL`/`OLLAMA_MODEL` (remote/lightweight-local).

## Backends at a glance

| Backend        | Entry point                                       | Deps                      | Where it runs                           | Best for                                   |
|----------------|---------------------------------------------------|---------------------------|-----------------------------------------|--------------------------------------------|
| `transformers` | [`llm_run.py`](llm_run.py) 📎                     | `requirements_llm.txt`    | Local, single GPU                       | Models ≤ 31 B (BnB 4-bit / AWQ / GGUF)     |
| `vllm`         | [`llm_run.py`](llm_run.py) 📎                     | `requirements_llm.txt`    | Local, single or multi-GPU              | Models ≥ 70 B, or any multi-GPU node       |
| `openrouter`   | [`openrouter_client.py`](openrouter_client.py) 📎 | `requirements_remote.txt` | Remote (OpenRouter API)                 | No local GPU; provider-routed data control |
| `ollama`       | [`ollama_client.py`](ollama_client.py) 📎         | `requirements_remote.txt` | Local, via `ollama serve` (CPU/any GPU) | Lightweight local runs, no heavy stack     |

`llm_run.py`'s `BACKEND` config value only switches between `transformers` and `vllm`;
OpenRouter and Ollama are separate CLI entry points. All four share the same quality filter,
context-window builder, archaeological Pydantic schema, and JSON validation — the local pair via
[`llm_utils.py`](llm_utils.py) 📎, the remote/lightweight-local pair via
[`llm_client_shared.py`](llm_client_shared.py) 📎 — so output records are directly comparable
across backends.

## Configuration (`llm_config.txt`)

All four entry points read [`llm_config.txt`](llm_config.txt) 📎 (override with `--config` on the
remote/lightweight-local clients, or a positional arg on `llm_run.py`).

```text
# ── Local (transformers/vLLM) ──────────────────────────────────────────────
MODEL_KEY=qwen-3.6-27b-it
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx   # gated models only: gemma-4-*, gemma-3-*, llama3.1-70b, llama4-*

INPUT_DIR=data_samples/DOC_LINE_LANG_CLASS
OUTPUT_DIR=data_samples/KW_PER_DOC_LLM
VOCAB_PATH=data_samples/teater_nested_vocab.json
PARADATA_DIR=paradata

INCLUDE_NON_TEXT=true
MIN_CHAR_COUNT=3
MIN_CHAR_NON_TEXT=8
MIN_ALPHA_RATIO_NON_TEXT=0.4

# BACKEND=vllm                  # auto-selected per model; override only to force a choice
# TENSOR_PARALLEL_SIZE=8        # vLLM only — GPUs to shard across
# GPU_MEMORY_UTILIZATION=0.88   # vLLM only
# VLLM_BATCH_SIZE=8             # vLLM only — lines per generate() call
# MAX_MODEL_LEN=16384           # cap context to reduce KV-cache pressure
# CPU_OFFLOAD_GB=70             # GB of weights kept in CPU RAM (needs vLLM ≥ 0.8.x)
# GUIDED_DECODING_BACKEND=xgrammar
# ENABLE_PREFIX_CACHING=false   # not recommended — reduces throughput

# ── Remote / lightweight-local (openrouter_client.py, ollama_client.py) ────
# OPENROUTER_MODEL=openai/gpt-4o-mini
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
# OLLAMA_MODEL=qwen2.5:7b
# OLLAMA_HOST=http://localhost:11434
```

> [!TIP]
> On startup, `llm_run.py` prints every effective inference parameter next to a label showing
> where it came from — `← llm_config.txt` (you set it), `(model default)` (from
> `MODEL_REGISTRY[...]["inference_defaults"]`), or `(global default)`. Check that log before
> adding overrides — most values are already covered by the model's own defaults.

Every knob on the remote/lightweight-local clients can also be passed as a CLI flag, which takes
precedence over `llm_config.txt` (see `--help` on either script for the full list — e.g.
`--provider-data-collection`, `--attach-as-file`, `--context-window`, `--max-retries`).

## Vocabulary Harvesting (`vocab_manager.py`)

Before running any backend, build the allowable vocabulary list. [`VocabularyManager`](vocab_manager.py) 📎 queries
the AMCR OAI-PMH endpoint for Czech–English term pairs, groups them into a thematic taxonomy, and
caches the result at `VOCAB_PATH` (default `data_samples/teater_nested_vocab.json`).

```bash
python3 vocab_manager.py
```

All four backends inject this vocabulary into their system prompt and constrain
`teater_category` to an enum built from it — a line/passage can only be tagged with a term that
actually exists in the thesaurus.

## Local Inference — `transformers` / `vLLM` (`llm_run.py`)

Reads every `*.csv` / `*.teitok.xml` file in `INPUT_DIR`, filters lines by quality, injects the
vocabulary and a sliding context window into the system prompt, and runs **constrained
decoding** — `lmformatenforcer` for `transformers`, native `xgrammar` guided decoding for `vllm`
— so the model cannot emit a `teater_category` outside the thesaurus or malformed JSON.

```bash
# Transformers backend (default — set BACKEND in llm_config.txt to switch)
python3 llm_run.py

# Custom config file
python3 llm_run.py my_config.txt
```

For multi-GPU runs, set `BACKEND=vllm`, `MODEL_KEY=<a large/MoE registry key>`, and
`TENSOR_PARALLEL_SIZE=<GPU count>` in `llm_config.txt`, then run the same command.

Output: `<stem>_enriched.json` per document, written to `OUTPUT_DIR_<model_suffix>/` — see
[Inputs and Outputs](#-inputs-and-outputs).

## Remote Inference — OpenRouter (`openrouter_client.py`)

The remote LLM-as-a-service backend (per #24: "explore file-attachment options and provider
routing for a local-only / no-logging data source"). Reuses the exact same Pydantic schema and
`validate_llm_output()` contract as the local backends, via `llm_client_shared.py`.

```bash
export OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
python3 openrouter_client.py --input data_samples/DOC_LINE_LANG_CLASS --model openai/gpt-4o-mini

# Data-sovereignty routing — restrict to providers that don't retain prompts/completions
python3 openrouter_client.py --input sample.csv --model <model> --provider-data-collection deny
```

> [!NOTE]
> `--provider-data-collection deny` restricts *routing*, not *licensing* — it does not by itself
> resolve which license applies to a given OpenRouter model's output. See the TODO in
> [`para_config.txt`](para_config.txt) 📎.

Two input modes, dispatched by file extension:

| Extension              | Mode           | One API call per… |
|------------------------|----------------|-------------------|
| `.csv`, `*.teitok.xml` | Line-level     | qualifying line   |
| `.md`, `.txt`          | Whole-document | document          |

`.md`/`.txt` input (typically rendered by [`api_util/xml_to_md.py`](api_util/xml_to_md.py) 📎 —
see [below](#document-level-input-api_utilxml_to_mdpy)) can optionally be sent as a file
attachment with `--attach-as-file` rather than inlined as message text; support for this varies
by model/provider and falls back silently to inlined text where unsupported.

## Lightweight Local — Ollama (`ollama_client.py`)

The lightweight local alternative to the heavy `transformers`/`vLLM` path: talks to a
locally-running `ollama serve` over HTTP instead of loading weights into this process.
Structured output uses Ollama's native `format` JSON-schema parameter (Ollama ≥ 0.5) — no
external constrained-decoding library needed.

```bash
ollama serve   # in a separate terminal, if not already running

python3 ollama_client.py --input data_samples/DOC_LINE_LANG_CLASS --model qwen2.5:7b
```

If the requested model isn't in `ollama list`, `ollama_client.py` triggers `POST /api/pull` and
streams progress before the first inference call (skip with `--skip-pull-check`). Same two
input-mode dispatch (CSV/TEITOK → line-level, `.md`/`.txt` → whole-document) as
`openrouter_client.py` above.

## Document-Level Input (`api_util/xml_to_md.py`)

Renders a whole TEITOK (`*.teitok.xml`) or raw ALTO XML document to Markdown or plain text, so an
entire document can be fed to an LLM as a single prompt — complementing the line-level CSV/TEITOK
row reader used by the per-line workflow above.

```bash
python3 api_util/xml_to_md.py sample.teitok.xml --format markdown --output sample.md
```

Output is page-sectioned (`## Page N` headings) so the whole-document system prompt's locator
instructions ("prefer including the nearest page heading above the located passage") resolve
against real anchors in the rendered text.

## 🖥 Model Registry

The built-in registry in `llm_utils.py` (shared, unmodified, with `atrium-nlp-enrich`) covers the
full range of supported local models. VRAM figures assume BnB 4-bit for `transformers` and
FP8/BF16 for `vllm`.

#### Single-GPU — `BACKEND=transformers` (or `BACKEND=vllm`)

| Registry key      | Model                         | Size       | Context | Est. VRAM | Notes                                                     |
|-------------------|-------------------------------|------------|---------|-----------|-----------------------------------------------------------|
| `qwen-3.6-27b-it` | Qwen/Qwen3.6-27B              | 27 B dense | 262 k   | ~18 GB    | **Default.** Best accuracy/VRAM ratio on a single GPU.    |
| `gemma-4-31b-it`  | google/gemma-4-31B-it         | 31 B dense | 256 k   | ~21 GB    | Highest single-GPU accuracy. Gated — `HF_TOKEN` required. |
| `qwen3-14b`       | OpenPipe/Qwen3-14B-Instruct   | 14 B dense | 128 k   | ~9 GB     | Good baseline.                                            |
| `qwen-3.5-9b-it`  | Qwen/Qwen3.5-9B               | 9 B dense  | 262 k   | ~6 GB     | Entry-level (8 GB VRAM).                                  |
| `qwen3-8b`        | Qwen/Qwen3-8B                 | 8 B dense  | 128 k   | ~16 GB    | BF16 (no 4-bit); straightforward baseline.                |
| `qwen2.5-14b-awq` | Qwen/Qwen2.5-14B-Instruct-AWQ | 14 B AWQ   | 128 k   | ~9 GB     | Pre-quantized; fast on NVIDIA GPUs.                       |
| `qwen2.5-7b`      | Qwen/Qwen2.5-7B-Instruct      | 7 B dense  | 32 k    | ~14 GB    | BF16; short context window.                               |
| `gemma-3-12b-it`  | google/gemma-3-12b-it         | 12 B dense | 128 k   | ~8 GB     | Good bilingual extraction. Gated.                         |

#### MoE / large — `BACKEND=vllm`

| Registry key           | Model                                         | Total / Active            | Context | Rec. TP | Notes                                                               |
|------------------------|-----------------------------------------------|---------------------------|---------|---------|---------------------------------------------------------------------|
| `gemma-4-26b-moe-gguf` | bartowski/google_gemma-4-26B-A4B-it-GGUF      | 4 B                       | 8 k     | 1       | GGUF/llama.cpp single-GPU fallback for `gemma-4-26b-moe`.           |
| `qwen-3.6-35b-moe`     | Qwen/Qwen3.6-35B-A3B                          | 35 B / 3 B                | 262 k   | 1       | Requires vLLM ≥ 0.8.x. Usually fits a single GPU.                   |
| `gemma-4-26b-moe`      | google/gemma-4-26B-A4B-it                     | 26 B / 4 B                | 256 k   | 2       | Gated.                                                              |
| `gemma-4-26b-moe-awq`  | google/gemma-4-26B-A4B-it                     | 26 B / 4 B                | 256 k   | 2       | AWQ-quantised variant of `gemma-4-26b-moe`. Gated.                  |
| `qwen3-235b-a22b-fp8`  | Qwen/Qwen3-235B-A22B-Instruct-2507-FP8        | 235 B / 22 B              | 128 k   | 2–8     | Requires vLLM ≥ 0.8.x. Native FP8 needs compute-capability ≥ 8.9.   |
| `qwen3-235b-a22b`      | Qwen/Qwen3-235B-A22B-Instruct-2507            | 235 B / 22 B              | 128 k   | 2       | BF16 variant — heavier than FP8.                                    |
| `deepseek-v3`          | deepseek-ai/DeepSeek-V3                       | 671 B MoE                 | 128 k   | 4–8     | FP8 official checkpoint. 4×80 GB minimum.                           |
| `llama4-maverick`      | meta-llama/Llama-4-Maverick-17B-128E-Instruct | 128 experts / 17 B active | 1 M     | 2       | Multimodal, 1 M token context. Gated. ⚠ needs ≥ 8× A100/H100 80 GB. |
| `llama3.1-70b`         | meta-llama/Meta-Llama-3.1-70B-Instruct        | 70 B dense                | 128 k   | 2       | Also works with `transformers` + 4-bit on 2×40 GB. Gated.           |

> [!TIP]
> **Automatic Prefix Caching (APC)** — enabled by default for the vLLM backend
> (`ENABLE_PREFIX_CACHING=true`). The system prompt (which embeds the full TEATER vocabulary) is
> computed once per run and its KV-cache is reused across every line in every document — the
> primary throughput multiplier, and it also removes the need to truncate the vocabulary to fit
> the token budget.

`CPU_OFFLOAD_GB` (vLLM ≥ 0.8.x, UVA zero-copy) lets weights that don't fit in GPU VRAM spill to
CPU RAM — see the config file's worked examples for `qwen3-235b-a22b-fp8` on 4×L40 vs. 8×A100.

## 📁 Inputs and Outputs

* **Input (local & remote/lightweight-local, line-level):** `INPUT_DIR/*.csv` or
  `*.teitok.xml` — expects `file_id`/`page_num`/`line_num`/`categ`/`quality_score`/`text` columns
  (CSV) or TEITOK's native `pb`/`lb`/`s` structure.
* **Input (remote/lightweight-local, document-level):** `.md`/`.txt`, typically from
  [`api_util/xml_to_md.py`](api_util/xml_to_md.py) 📎.
* **Output:** `<OUTPUT_DIR>_<model_suffix>/*_enriched.json` — one file per document.
* **Abort sidecar:** `*_enriched.abort.json`, written only when a document is abandoned after 10
  consecutive inference errors — the canonical signal that the JSON output holds partial results.

**Example line-level output record:**
```json
{
  "file_id": "CTX195603828",
  "page": 1,
  "line": 14,
  "categ": "Text",
  "quality_score": 0.98,
  "original_text": "Výzkum odhalil základy gotického kostela ze 14. století.",
  "enrichment": {
    "extracted_keywords_cs": ["základy", "gotický kostel"],
    "extracted_keywords_en": ["foundations", "gothic church"],
    "teater_category": "kostel",
    "confidence_score": 0.95
  }
}
```

**Example whole-document output record** (`run_document_level`, remote/lightweight-local only):
```json
{
  "file_id": "CTX195603828",
  "locator": "základy gotického kostela ze 14. století",
  "enrichment": {
    "extracted_keywords_cs": ["základy", "gotický kostel"],
    "extracted_keywords_en": ["foundations", "gothic church"],
    "teater_category": "kostel",
    "confidence_score": 0.95
  }
}
```

**Abort sidecar format (`*_enriched.abort.json`):**
```json
{
  "aborted": true,
  "abort_reason": "10 consecutive inference errors",
  "processed_before_abort": 42,
  "errors_before_abort": 10,
  "timestamp_utc": "2026-05-20T09:14:33"
}
```

## 📐 Document Understanding benchmark (`sample_stratify.py` + `bench_compare.py`)

The head-to-head evaluation harness decided in hub issue
[#22](https://github.com/ufal/atrium-project/issues/22): out-of-the-box VLM/OCR models vs. the
legacy ABBYY/ALTO pipeline, scored per quality tier on an in-domain gold set. Three pieces, all
torch-free (`eval_metrics.py` is pure stdlib):

**1. Sample pages, stratified by OCR quality** — consumes the per-page stats produced by
`atrium-alto-postprocess` (`samples_page_stats.csv`, or a `DOC_LINE_CATEG/` directory aggregated
on the fly), buckets pages into difficulty tiers (`clean` / `degraded` / `hard` / `text_poor`),
and writes an annotation manifest with a deterministic 80/10/10 train/dev/test split:

```bash
python sample_stratify.py --page-stats samples_page_stats.csv --n 200 --output docu_sample_manifest.csv
python sample_stratify.py --lines-dir ../atrium-alto-postprocess/data_samples/DOC_LINE_CATEG --n 40
```

**2. Annotate gold transcriptions** — the gold directory mirrors the
`atrium-alto-postprocess` `PAGE_TXT*` layout, one UTF-8 plain-text file per manifest page in
reading order (whitespace is normalized before scoring, so line breaks are free):

```
gold/
└── CTX192100040/
    ├── CTX192100040-1.txt              # required: full-page transcription
    └── CTX192100040-1.entities.tsv     # optional: TYPE<TAB>surface text (CNEC 2.0 / TEATER)
```

Entity sidecars are scored only with `--entities`; a page joins the entity aggregates only when
the *gold* sidecar exists (a missing hypothesis sidecar counts as zero predicted entities).
Table scoring (TEDS) is deferred until `table_teds.py` is vendored.

**3. Run the comparison** — each named `--pred` directory is one "model"; the `PAGE_TXT*`
outputs of `atrium-alto-postprocess` (alto-tools / LayoutReader / GLM-4v) can be consumed
directly, alongside any VLM transcriptions written in the same layout:

```bash
python bench_compare.py --manifest docu_sample_manifest.csv \
    --gold data/gold \
    --pred alto=../atrium-alto-postprocess/data_samples/PAGE_TXT \
           layoutreader=../atrium-alto-postprocess/data_samples/PAGE_TXT_LR \
           glm=../atrium-alto-postprocess/data_samples/PAGE_TXT_LLM \
    --split test --output-dir bench_results
```

Outputs in `--output-dir` (deterministic, byte-identical across reruns):

| File                   | Content                                                                                                                                                                               |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `page_scores.csv`      | long format — one row per model × page: CER, WER, NED, char counts, optional entity P/R/F1                                                                                            |
| `aggregate_scores.csv` | per model × tier (+ `overall`): macro CER/WER/NED, micro-pooled entity P/R/F1, page/missing counts                                                                                    |
| `report.md`            | model-comparison tables (overall + per-tier), best values bolded — the DU analogue of the [page-classification](https://github.com/ufal/atrium-page-classification) comparison tables |


| Model            | Pages | CER (%)  | WER (%)  | NED       |
|------------------|-------|----------|----------|-----------|
| **layoutreader** | 7     | **0.00** | **0.00** | **0.000** |
| mock-vlm         | 6     | 1.41     | 8.36     | 0.014     |


Missing prediction files are skipped and counted per model (`--missing error` to fail instead);
missing gold files exclude the page for all models. Every run drops a paradata JSON
(see [Paradata Logs](#paradata-logs)). Both scripts also read an optional shared INI config
(`--config config_docu.txt`) with `[STRATIFY]` and `[BENCHMARK]` sections:

```ini
[BENCHMARK]
MANIFEST = docu_sample_manifest.csv
GOLD_DIR = data/gold
PRED_DIRS = alto=../alto/PAGE_TXT, layoutreader=../alto/PAGE_TXT_LR
SPLIT = test
ENTITIES = false
MISSING = skip
OUTPUT_DIR = bench_results
```

## 🐳 Docker

Three build targets, layered so each installs only the deps it needs:

```bash
# Local, transformers/vLLM — heavy GPU stack (torch, transformers, vLLM, bitsandbytes)
docker build --target llm -t atrium-llm-enrich:llm .
docker run --gpus all -v "$PWD/data_samples:/app/data_samples" atrium-llm-enrich:llm

# Remote (OpenRouter) or lightweight-local (Ollama) — light client deps only
docker build --target remote -t atrium-llm-enrich:remote .
docker run -e OPENROUTER_API_KEY atrium-llm-enrich:remote openrouter_client.py --input sample.csv --model <model>
docker run atrium-llm-enrich:remote ollama_client.py --host http://host.docker.internal:11434 --input sample.csv --model qwen2.5:7b
```

> [!NOTE]
> A `docker-compose.gpu.yaml` with GPU reservations (matching `atrium-nlp-enrich`'s pattern) is
> not yet present in this repo — currently plain `docker run --gpus all` (above) or manual
> Compose GPU device reservations are the way to run the `llm` image with GPU access.

## Paradata Logs

Every entry point records structured provenance metadata through
[`atrium_paradata.py`](atrium_paradata.py) 📎, dropped into `PARADATA_DIR` (default `paradata/`)
as:

```terminaloutput
YYMMDD-HHmmss_llm-enrich.json
```

The log captures the run ID, execution duration, the full `llm_config.txt` snapshot (backend,
model, quality-filter settings), input/output counts, skipped files with reasons, and — for
local backends — token throughput (`total input tokens`, `avg tok/s`). The effective license
block is currently a documented open TODO — see [`para_config.txt`](para_config.txt) 📎 for why
(OpenRouter/Ollama/local model licenses vary per `MODEL_KEY`/provider rather than being a single
fixed component like `atrium-nlp-enrich`'s NER models).

> [!TIP]
> While a run is in progress, `atrium_paradata.py` keeps intermediate state in
> `PARADATA_DIR/.state_<runid>_llm-enrich.json` — plain JSON, inspectable if a run is interrupted
> unexpectedly. It is removed automatically on successful completion.

---

## Acknowledgements 🙏

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^1] 🔗

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4] 💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
- **LLM engine copied from** `atrium-nlp-enrich` [^2] (ATRIUM issue #24 [^3])
- **Frameworks used**:
  - HuggingFace **Transformers** + **bitsandbytes** (single-GPU 4-bit inference)
  - **vLLM** + **xgrammar** (multi-GPU, native guided JSON decoding)
  - **OpenRouter** [^5] (remote LLM-as-a-service)
  - **Ollama** [^6] (lightweight local inference server)
  - UFAL **flexiconv** [^8] (optional txt/pdf/docx/html/md → TEITOK conversion)

**©️ 2026 UFAL & ATRIUM**

[^1]: https://github.com/ufal/atrium-llm-enrich
[^2]: https://github.com/ufal/atrium-nlp-enrich
[^3]: https://github.com/ufal/atrium-project/issues/24
[^4]: https://atrium-research.eu/
[^5]: https://openrouter.ai/
[^6]: https://ollama.com/
[^7]: https://ufal.mff.cuni.cz/
[^8]: https://github.com/ufal/flexiconv