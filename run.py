"""
Entry point for the AI Travel Planner.

Run:  python run.py
      ./travel.sh
"""

# ── Silence all third-party noise BEFORE any library import ──────────────────
# HuggingFace / sentence-transformers check these env-vars at import time, so
# they must be set here — setting them inside semantic_cache.py is too late.
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import logging
import warnings

warnings.warn = lambda *args, **kwargs: None  # suppress all third-party warnings
logging.disable(logging.CRITICAL)             # suppress all log output to the terminal

from src.config.settings import ConfigurationError, settings
from src.main import run

if __name__ == "__main__":
    try:
        settings.validate_startup()
    except ConfigurationError as exc:
        print(f"\n[Marco] Configuration error:\n{exc}\n")
        raise SystemExit(1)
    run()
