from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv


# Ensure `import src.*` works when pytest runs from repo root in CI/local.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure tests (especially live tests) can read API keys from .env.
load_dotenv(REPO_ROOT / ".env")
