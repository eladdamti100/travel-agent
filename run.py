"""
Entry point for the AI Travel Planner.

Run:  python run.py              — start the interactive terminal UI
      python run.py --check      — validate config and dependencies, then exit
"""

# ── Silence all third-party noise BEFORE any library import ──────────────────
# HuggingFace / sentence-transformers check these env-vars at import time, so
# they must be set here — setting them inside individual modules is too late.
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import logging
import sys
import warnings

warnings.warn = lambda *args, **kwargs: None  # suppress all third-party warnings
logging.disable(logging.CRITICAL)             # suppress log output to the terminal

from src.config.settings import ConfigurationError, settings


def _run_health_check() -> int:
    """
    Validates configuration and key dependencies.  Returns 0 on success, 1 on failure.
    Called by `python run.py --check`.
    """
    checks_passed = 0
    checks_failed = 0

    def _ok(label: str, detail: str = "") -> None:
        nonlocal checks_passed
        checks_passed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [OK]  {label}{suffix}")

    def _fail(label: str, detail: str = "") -> None:
        nonlocal checks_failed
        checks_failed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [FAIL] {label}{suffix}")

    print("\nMarco - startup health check\n" + "-" * 40)

    # 1. Required API key
    try:
        settings.validate_startup()
        _ok("LLM provider key", f"provider={settings.llm_provider} model={settings.resolved_model_name}")
    except ConfigurationError as exc:
        _fail("LLM provider key", str(exc).replace("\n", " "))

    # 2. SQLite travel DB
    try:
        if settings.travel_db_path.exists():
            _ok("Travel database", str(settings.travel_db_path))
        else:
            _fail("Travel database", f"not found at {settings.travel_db_path}  — run: python -m src.utils.db_init")
    except Exception as exc:
        _fail("Travel database", str(exc))

    # 3. Semantic cache dir
    try:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _ok("Cache directory", str(settings.cache_dir))
    except Exception as exc:
        _fail("Cache directory", str(exc))

    # 4. Embedding model (lazy-load check only — no download triggered)
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        _ok("Embedding model package (sentence-transformers)")
    except ImportError as exc:
        _fail("Embedding model package", str(exc))

    # 5. Optional web API keys
    print("\n  Optional web API keys:")
    opt_keys = [
        ("OPENCAGE_API_KEY",     settings.opencage_api_key),
        ("TICKETMASTER_API_KEY", settings.ticketmaster_api_key),
        ("EXCHANGERATE_API_KEY", settings.exchangerate_api_key),
        ("TAVILY_API_KEY",       settings.tavily_api_key),
    ]
    for name, val in opt_keys:
        if val and not val.startswith("your_"):
            print(f"    [SET]   {name}")
        else:
            print(f"    [UNSET] {name}  (fallback data will be used)")

    # Summary
    print("\n" + "-" * 40)
    if checks_failed:
        print(f"  {checks_passed} passed, {checks_failed} failed — fix the issues above and re-run.\n")
        return 1
    print(f"  All {checks_passed} checks passed. Marco is ready.\n")
    return 0


from src.main import run

if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(_run_health_check())

    try:
        settings.validate_startup()
    except ConfigurationError as exc:
        print(f"\n[Marco] Configuration error:\n{exc}\n")
        raise SystemExit(1)
    run()
