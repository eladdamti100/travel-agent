"""
Application settings — single source of truth for all configuration.

All env vars, thresholds, model names, and DB paths live here.
Other modules import from this module instead of calling os.getenv() directly.

Usage:
    from src.config.settings import settings

    api_key = settings.groq_api_key
    threshold = settings.cache_hit_threshold

Startup validation:
    Call settings.validate_startup() in run.py before the first user request.
    It raises ConfigurationError with a clear message listing every missing key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when required environment variables are missing at startup."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM provider ─────────────────────────────────────────────────────────
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None

    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # ── Optional web API keys (all have graceful fallbacks) ──────────────────
    opencage_api_key: Optional[str] = None
    ticketmaster_api_key: Optional[str] = None
    exchangerate_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None

    lakera_api_key: Optional[str] = None
    google_safe_browsing_key: Optional[str] = None

    # ── Observability (LangSmith) ─────────────────────────────────────────────
    # Set LANGSMITH_API_KEY and LANGSMITH_TRACING=true to enable tracing.
    langsmith_api_key: Optional[str] = None
    langsmith_tracing: bool = False
    langsmith_project: str = "marco-travel-planner"

    # ── Semantic cache ────────────────────────────────────────────────────────
    cache_hit_threshold: float = 0.85
    cache_ttl_days_db: int = 30
    cache_ttl_days_web: int = 3
    cache_max_ttl_days: int = 365
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Graph / planner limits ────────────────────────────────────────────────
    max_tool_calls: int = 8
    max_critic_attempts: int = 2
    max_hitl_edit_attempts: int = 3
    enrichment_timeout_seconds: float = 10.0
    planner_timeout_seconds: float = 90.0

    # ── Demo mode — set DEMO_MODE=true in .env for fast, reliable presentations ─
    # Skips slow live web agents; all results come from the local DB (instant).
    demo_mode: bool = False

    # ── DB paths (resolved at import time) ───────────────────────────────────
    cache_dir: Path = Path.home() / ".cache" / "travel-agent"

    @property
    def checkpoints_db_path(self) -> Path:
        return self.cache_dir / "checkpoints.db"

    @property
    def semantic_cache_db_path(self) -> Path:
        return self.cache_dir / "semantic_cache.db"

    @property
    def travel_db_path(self) -> Path:
        return Path(__file__).parent.parent.parent / "data" / "travel_agency.db"

    # ── Derived model name ────────────────────────────────────────────────────
    @property
    def resolved_model_name(self) -> str:
        if self.llm_model:
            return self.llm_model
        defaults = {
            "groq": "llama-3.1-8b-instant",
            "gemini": "gemini-2.5-flash",
        }
        return defaults.get(self.llm_provider.lower(), "gemini-2.5-flash")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("llm_provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        v = v.lower()
        if v not in ("groq", "gemini"):
            raise ValueError(f"llm_provider must be 'groq' or 'gemini', got '{v}'")
        return v

    @field_validator("cache_hit_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"cache_hit_threshold must be between 0 and 1, got {v}")
        return v

    # ── Startup check ─────────────────────────────────────────────────────────
    def validate_startup(self) -> None:
        """
        Raise ConfigurationError if any required keys are absent.

        Call this once in run.py before the graph starts.
        Web API keys are optional — their absence is noted as warnings, not errors.
        """
        missing: list[str] = []

        if self.llm_provider == "groq" and not self.groq_api_key:
            missing.append("GROQ_API_KEY (required when LLM_PROVIDER=groq)")
        if self.llm_provider == "gemini" and not self.google_api_key:
            missing.append("GOOGLE_API_KEY (required when LLM_PROVIDER=gemini)")

        if missing:
            raise ConfigurationError(
                "Missing required environment variables:\n"
                + "\n".join(f"  • {m}" for m in missing)
                + "\n\nCreate a .env file in the project root and set the above keys."
            )

        optional_warnings: list[str] = []
        for key, label in [
            (self.opencage_api_key, "OPENCAGE_API_KEY"),
            (self.ticketmaster_api_key, "TICKETMASTER_API_KEY"),
            (self.exchangerate_api_key, "EXCHANGERATE_API_KEY"),
            (self.tavily_api_key, "TAVILY_API_KEY"),
        ]:
            if not key or key.startswith("your_"):
                optional_warnings.append(label)

        if optional_warnings:
            from src.utils.logger import get_logger
            _log = get_logger("settings")
            _log.info(
                "Optional web API keys not configured (static fallbacks active): %s",
                ", ".join(optional_warnings),
            )

    def configure_tracing(self) -> bool:
        """
        Enable LangSmith tracing when LANGSMITH_TRACING=true and a key is set.

        Sets the LangChain environment variables that the SDK checks at runtime.
        Returns True if tracing was enabled, False otherwise.
        Call once in run.py after validate_startup().
        """
        import os as _os
        if not self.langsmith_tracing or not self.langsmith_api_key:
            return False

        _os.environ["LANGCHAIN_TRACING_V2"] = "true"
        _os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        _os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project

        from src.utils.logger import get_logger
        get_logger("settings").info(
            "tracing. provider=langsmith project=%s enabled=True",
            self.langsmith_project,
        )
        return True


# Module-level singleton — import this everywhere instead of re-instantiating.
settings = Settings()
