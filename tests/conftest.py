"""
tests/conftest.py
=================
Shared pytest fixtures and sys.path wiring for atrium-llm-enrich unit tests.

sys.path is patched here (once, at collection time) so that every test module
can import from both the repo root (``llm_client_shared.py``,
``atrium_paradata.py``) and the ``api_util/`` subdirectory
(``xml_to_md``, ``teitok_read``).
"""

import sys
from pathlib import Path

# ── path wiring ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "api_util"))