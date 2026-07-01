## 2026-07-01
- **#24** — `atrium-llm-enrich` bootstrapped (owner action, `K4TEL/atrium-llm-enrich`, not yet
  transferred to `ufal`) and built out to match `plans/24.plan.md`'s proposed layout end to end:
  engine copied byte-identical from `nlp-enrich`; `openrouter_client.py`, `ollama_client.py`,
  `api_util/xml_to_md.py`, and the shared `llm_client_shared.py` front-end added; a broken
  post-bootstrap `Dockerfile` (referenced dropped files: `run_pipeline.py`, four `api_*.sh`,
  `service/`, `requirements-test.txt`) rewritten around the repo's actual `base → remote → llm`
  stages; `README.md` replaced (was the strategy doc committed verbatim); `tests/`,
  `.github/workflows/` (8 workflows + dependabot), `docker-compose.yaml`/`docker-compose.gpu.yaml`,
  and `CONTRIBUTING.md` added. See `digests/24.digest.md` for the full state and open items
  (org transfer, per-model licensing TODO, live-inference verification still pending).