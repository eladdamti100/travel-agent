"""
AI Validator — LLM-powered security guardrail using Groq.

Replaces regex-based pattern matching with a dedicated LLM classifier
that understands intent, context, and creative injection attempts that
regex cannot catch.

Architecture:
1. ai_validate()  → calls Groq llama-3.1-8b-instant (fast, free tier)
2. Returns the same ValidationResult interface as the hardcoded validator
3. nodes.py calls this first; if Groq is unavailable it falls back to
the hardcoded InputValidator automatically

Why Groq for validation (not the main agent model):
- Always available even when main provider is Gemini
- llama-3.1-8b-instant: ~200ms response, handles classification well
- Separate concern — validation is independent of planning
- Free tier: 14,400 req/day — plenty for a validator
"""

import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

load_dotenv()

from src.agents.validator import ValidationResult, _REJECTION_MESSAGES
from src.utils.logger import get_logger
from src.prompts.loader import get_prompt

logger = get_logger("ai_validator")


# ── Groq client — created once at module level ────────────────────────────────
_groq_model: Optional[ChatGroq] = None


def _get_groq_model() -> Optional[ChatGroq]:
    """
    Returns a cached Groq client for validation.
    Uses llama-3.1- 8b -instant: fast (~ 200ms ), accurate for classification,
    generous free tier (14,400 req/day).
    Returns None if GROQ_API_KEY is not configured.
    """
    global _groq_model
    if _groq_model is not None:
        return _groq_model

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key.startswith("your_"):
        return None

    _groq_model = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=150,
        timeout=8,
    )
    return _groq_model


# ── Main AI validation function ───────────────────────────────────────────────

def ai_validate(message: str) -> Optional[ValidationResult]:
    """
    Validates a user message using Groq LLM.

    Sends the message to llama-3.1- 8b -instant with a comprehensive policy
    prompt. The model returns a JSON verdict that is parsed into a
    ValidationResult — the same interface used by the hardcoded validator.

    Returns:
    ValidationResult — if Groq responded successfully
    None             — if Groq is unavailable or timed out (caller falls back
    to hardcoded InputValidator)
    """
    model = _get_groq_model()
    if model is None:
        logger.warning("AI validator: GROQ_API_KEY not set — falling back to hardcoded validator.")
        return None

    try:
        response = model.invoke([
            SystemMessage(content=get_prompt("validator_prompt")),
            HumanMessage(content=f"Validate this user message:\n\n\"{message}\""),
        ])

        raw = response.content.strip()

        # Strip markdown code fences if the model wrapped its JSON
        if "```" in raw:
            raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)

        approved: bool = data.get("approved", True)
        verdict: str   = data.get("verdict", "APPROVED")
        reason: str    = data.get("reason", "")

        rejection_msg = _REJECTION_MESSAGES.get(verdict, _REJECTION_MESSAGES["BLOCKED_SCOPE"])

        logger.info("AI validator: verdict=%s reason=%s", verdict, reason)

        return ValidationResult(
            approved=approved,
            verdict=verdict,
            reason=reason,
            rejection_message=rejection_msg,
        )

    except json.JSONDecodeError as e:
        logger.error("AI validator: JSON parse error — %s | raw=%s", e, raw[:200])
        return None
    except Exception as e:
        logger.error("AI validator: unexpected error — %s", e)
        return None