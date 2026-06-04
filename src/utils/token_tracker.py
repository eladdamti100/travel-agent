"""
Token and cost tracker for LLM calls.

Usage:
    from src.utils.token_tracker import log_token_usage

    response = await model.ainvoke(messages)
    log_token_usage(response, call_site="planner.notes_section")
"""

from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

logger = get_logger("token_tracker")

# Approximate cost per 1 000 tokens (USD).
# These are estimates — update when provider pricing changes.
_COST_PER_1K: dict[str, dict[str, float]] = {
    # Groq (free tier / paid)
    "llama-3.1-8b-instant":  {"input": 0.00005,  "output": 0.00008},
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    # Gemini
    "gemini-2.5-flash":      {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash":      {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro":        {"input": 0.00125,  "output": 0.005},
}


def log_token_usage(response: Any, call_site: str = "unknown") -> dict[str, int | float]:
    """
    Extract token usage from an LLM response and log it.

    Returns a dict with input_tokens, output_tokens, total_tokens, estimated_cost_usd.
    Returns an empty dict if usage metadata is unavailable (never raises).
    """
    try:
        usage = _extract_usage(response)
        if not usage:
            return {}

        model_name = _get_model_name(response)
        costs = _COST_PER_1K.get(model_name, {})
        estimated_cost = (
            usage.get("input_tokens", 0) * costs.get("input", 0) / 1000
            + usage.get("output_tokens", 0) * costs.get("output", 0) / 1000
        )
        usage["estimated_cost_usd"] = round(estimated_cost, 8)

        logger.info(
            "token_usage. call_site=%s model=%s input=%d output=%d total=%d cost_usd=%.6f",
            call_site,
            model_name,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("total_tokens", 0),
            estimated_cost,
        )
        return usage
    except Exception as exc:
        logger.debug("token_usage extraction failed for %s: %s", call_site, exc)
        return {}


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token counts from any LangChain response object."""
    # LangChain standardised field (works for most providers in newer versions)
    meta = getattr(response, "usage_metadata", None)
    if meta and isinstance(meta, dict):
        return {
            "input_tokens":  meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "total_tokens":  meta.get("total_tokens", 0),
        }

    # Groq / OpenAI-compatible: token_usage in response_metadata
    resp_meta = getattr(response, "response_metadata", None) or {}
    token_usage = resp_meta.get("token_usage") or resp_meta.get("usage")
    if token_usage:
        return {
            "input_tokens":  token_usage.get("prompt_tokens", 0),
            "output_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens":  token_usage.get("total_tokens", 0),
        }

    return {}


def _get_model_name(response: Any) -> str:
    """Best-effort extraction of the model name from the response."""
    meta = getattr(response, "response_metadata", None) or {}
    return (
        meta.get("model")
        or meta.get("model_name")
        or meta.get("model_id")
        or getattr(response, "model", None)
        or "unknown"
    )
