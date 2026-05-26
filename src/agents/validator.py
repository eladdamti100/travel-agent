"""
Unified validator — hardcoded rule-based guardrail + Groq AI classifier.

Three-stage pipeline (fastest first):
  1. Instant regex — harm, injection, city, off-topic (no LLM cost)
  2. Travel keyword fast-approve — skips the Groq call for obvious travel messages
  3. Groq LLM check — for ambiguous messages only (~200 ms, llama-3.1-8b-instant)

HITL safety:
  When the planner is awaiting user clarification (is_hitl=True), only harm and
  injection are checked.  Short factual answers like "TLV", "7 days", "$2000",
  or "Israeli passport" are fast-approved before any regex scan.

Hardened injection detection:
  Catches classic jailbreaks, roleplay persona switches, language-change tricks,
  "unfiltered mode" attempts, and hypothetical-framing bypasses.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from src.agents.preferences_memory_agent import _is_recall_request
from src.agents.validator_patterns import (
    HARM_PATTERNS_RAW,
    INJECTION_PATTERNS_RAW,
    KNOWN_UNSUPPORTED_CITIES,
    OFF_TOPIC_PATTERNS_RAW,
    STRONG_TRAVEL_SIGNALS,
    SUPPORTED_CITIES,
    TRAVEL_KEYWORDS,
    VISA_COUNTRY_NAMES,
)
from src.prompts.loader import get_prompt
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger("validator")


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Result returned by every stage of the input validation pipeline."""

    approved: bool
    verdict: str          # APPROVED | BLOCKED_HARM | BLOCKED_INJECTION | BLOCKED_SCOPE | BLOCKED_CITY
    reason: str
    rejection_message: str


# ── Rejection messages shown to the user ─────────────────────────────────────

_REJECTION_MESSAGES = {
    "BLOCKED_HARM": (
        "I'm sorry, but I'm unable to process your request as it appears to violate "
        "our usage guidelines. I'm Marco, a travel planning assistant here to help you "
        "plan wonderful trips. Please keep our conversation respectful and travel-related. "
        "Feel free to ask me about flights, hotels, or activities!"
    ),
    "BLOCKED_INJECTION": (
        "I noticed your message contains instructions trying to change my behaviour. "
        "I'm Marco, your travel planning assistant, and I can only help with "
        "flights, hotels, activities, and trip planning."
    ),
    "BLOCKED_SCOPE": (
        "That's outside what I can help with. I'm Marco, a travel planning assistant. "
        "Ask me about flights, hotels, activities, or trip costs for "
        "Paris, London, Tokyo, New York, or Berlin."
    ),
    "BLOCKED_CITY": (
        "Sorry, that destination isn't in my database yet. "
        "I currently support: Paris, London, Tokyo, New York, and Berlin. "
        "Would you like to plan a trip to one of those instead?"
    ),
}


# ── HITL safety — fast-approve short factual clarification answers ─────────────

# Patterns that unambiguously identify a HITL clarification answer.
_HITL_SAFE_PATTERNS = [
    # 3-letter IATA airport codes: TLV, JFK, LHR, CDG, NRT, …
    re.compile(r"\b[A-Z]{3}\b"),
    # Nationalities / passport countries
    re.compile(
        r"\b(israeli|american|british|french|german|japanese|canadian|australian|"
        r"italian|spanish|dutch|swedish|norwegian|danish|swiss|austrian|belgian|"
        r"portuguese|greek|polish|hungarian|czech|eu|european|passport)\b",
        re.IGNORECASE,
    ),
    # Trip duration
    re.compile(r"\b\d+\s*(days?|nights?|weeks?|months?)\b", re.IGNORECASE),
    # Budget / money amounts
    re.compile(r"(\$|€|£|¥)?\s*\d[\d,]*(\.\d+)?\s*(dollars?|usd|eur|gbp|jpy|euros?|pounds?)?\b", re.IGNORECASE),
    # Number of travellers
    re.compile(r"\b\d+\s*(people|persons?|travelers?|travellers?|passengers?|adults?|kids?|children)\b", re.IGNORECASE),
]

_HITL_MAX_LENGTH = 100  # messages longer than this are not considered "short factual answers"


def _is_hitl_safe(message: str) -> bool:
    """
    Returns True for short factual HITL answers that are never harmful.

    Checks length first (HITL answers are concise), then scans for travel-detail
    patterns like airport codes, nationalities, durations, and budgets.
    """
    if len(message.strip()) > _HITL_MAX_LENGTH:
        return False
    for pattern in _HITL_SAFE_PATTERNS:
        if pattern.search(message):
            return True
    return False


# ── Extra hardened injection patterns (appended to INJECTION_PATTERNS_RAW) ────

_EXTRA_INJECTION_PATTERNS_RAW = [
    # Language/style persona switch ("speak like a pirate / gangster / thug")
    r"(respond|speak|talk|write|communicate|reply|answer)\s+(in\s+the\s+style\s+of|like\s+a?|as\s+a?)\s+"
    r"(pirate|gangster|thug|gang|ghetto|street|hacker|criminal|villain|robot|evil|nazi|terrorist)",
    # "Without [any] restrictions / filters / safety"
    r"(without|no)\s+(\w+\s+)?(restrictions?|limits?|filters?|safety|guardrails?|rules?|guidelines?)",
    # Unfiltered / unrestricted mode requests
    r"(unfiltered|uncensored|unrestricted|unsafe)\s*(version|mode|response|ai|llm|model|output)?",
    # Hypothetical / fictional framing bypass
    r"hypothetically\s+(speaking|if|,)\b",
    r"in\s+a\s+(fictional|hypothetical|alternate|parallel|simulated)\s+(world|universe|scenario|reality|story)",
    # "For training / research / testing purposes" bypass attempt
    r"for\s+(training|testing|research|demonstration|educational)\s+purposes?\s*(only|,|\.|$)",
    # Direct command injection
    r"execute\s+(the\s+)?(following|this|next)\s+(command|instruction|script|order)",
    # Token smuggling ("[SYSTEM]", "###Instruction:", etc.)
    r"(\[SYSTEM\]|\[INST\]|###\s*instruction|<\|system\|>|<\|im_start\|>)",
    # "From now on you will" / "starting now you are"
    r"(from\s+now\s+on|starting\s+now|going\s+forward)\s+(you\s+)?(will|are|must|should|have\s+to)",
    # "Your true self / inner self / real personality"
    r"your\s+(true|real|inner|hidden|secret|actual)\s+(self|personality|identity|nature|purpose|instructions?)",
]


# ── Main validator class ──────────────────────────────────────────────────────

class InputValidator:
    """
    Stateless hardcoded validator.  All patterns compile once on first use.
    Public surface:
      InputValidator.validate(message)       → ValidationResult
      InputValidator.detect_harm(message)    → (verdict, reason) | None
      InputValidator.is_clearly_travel(msg)  → bool
    """

    SUPPORTED_CITIES = SUPPORTED_CITIES
    KNOWN_UNSUPPORTED_CITIES = KNOWN_UNSUPPORTED_CITIES

    _HARM_PATTERNS_RAW = HARM_PATTERNS_RAW
    _INJECTION_PATTERNS_RAW = INJECTION_PATTERNS_RAW + _EXTRA_INJECTION_PATTERNS_RAW
    _OFF_TOPIC_PATTERNS_RAW = OFF_TOPIC_PATTERNS_RAW
    _STRONG_TRAVEL_SIGNALS = STRONG_TRAVEL_SIGNALS
    _TRAVEL_KEYWORDS = TRAVEL_KEYWORDS
    _VISA_COUNTRY_NAMES = VISA_COUNTRY_NAMES

    # Compiled pattern caches
    _harm_compiled: Optional[list] = None
    _injection_compiled: Optional[list] = None
    _off_topic_compiled: Optional[list] = None

    @classmethod
    def _harm_patterns(cls) -> list:
        if cls._harm_compiled is None:
            cls._harm_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._HARM_PATTERNS_RAW
            ]
        return cls._harm_compiled

    @classmethod
    def _injection_patterns(cls) -> list:
        if cls._injection_compiled is None:
            cls._injection_compiled = [
                re.compile(p, re.IGNORECASE) for p in cls._INJECTION_PATTERNS_RAW
            ]
        return cls._injection_compiled

    @classmethod
    def _off_topic_patterns(cls) -> list:
        if cls._off_topic_compiled is None:
            cls._off_topic_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._OFF_TOPIC_PATTERNS_RAW
            ]
        return cls._off_topic_compiled

    @staticmethod
    def detect_harm(message: str) -> Optional[tuple]:
        """
        Scans for harmful or violating content.
        Returns (verdict, reason) if harm detected, None if clean.
        """
        for pattern, label in InputValidator._harm_patterns():
            if pattern.search(message):
                return ("BLOCKED_HARM", f"Harmful content detected: {label}")
        return None

    @classmethod
    def validate(cls, message: str) -> ValidationResult:
        """Full validation pipeline: harm → injection → city → off-topic."""
        msg_lower = message.lower().strip()

        # 0. Harm (highest priority)
        harm = cls.detect_harm(message)
        if harm:
            verdict, reason = harm
            return ValidationResult(
                approved=False,
                verdict=verdict,
                reason=reason,
                rejection_message=_REJECTION_MESSAGES["BLOCKED_HARM"],
            )

        # 1. Injection
        for pattern in cls._injection_patterns():
            if pattern.search(message):
                return ValidationResult(
                    approved=False,
                    verdict="BLOCKED_INJECTION",
                    reason="Prompt injection attempt detected.",
                    rejection_message=_REJECTION_MESSAGES["BLOCKED_INJECTION"],
                )

        # 2. City
        detected = cls._detect_city(msg_lower)
        if detected and detected not in cls.SUPPORTED_CITIES:
            return ValidationResult(
                approved=False,
                verdict="BLOCKED_CITY",
                reason=f"Unsupported destination: {detected}.",
                rejection_message=(
                    f"Sorry, {detected.title()} isn't in my database yet. "
                    "I currently support: Paris, London, Tokyo, New York, and Berlin. "
                    "Would you like to plan a trip to one of those instead?"
                ),
            )

        # 3. Off-topic (only when no travel keyword present)
        has_travel_keyword = any(kw in msg_lower for kw in cls._TRAVEL_KEYWORDS)
        if not has_travel_keyword:
            for pattern, topic in cls._off_topic_patterns():
                if pattern.search(message):
                    return ValidationResult(
                        approved=False,
                        verdict="BLOCKED_SCOPE",
                        reason=f"Off-topic request: {topic}.",
                        rejection_message=_REJECTION_MESSAGES["BLOCKED_SCOPE"],
                    )

        return ValidationResult(
            approved=True,
            verdict="APPROVED",
            reason="Valid travel-related request.",
            rejection_message="",
        )

    @classmethod
    def is_clearly_travel(cls, message: str) -> bool:
        """
        Returns True when the message unambiguously relates to travel.
        Used to skip the Groq LLM call for obvious requests.
        """
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in cls._STRONG_TRAVEL_SIGNALS)

    @classmethod
    def _detect_city(cls, msg_lower: str) -> Optional[str]:
        for city in cls.SUPPORTED_CITIES:
            if city in msg_lower:
                return city
        for country in cls._VISA_COUNTRY_NAMES:
            if country in msg_lower:
                return None
        for city in cls.KNOWN_UNSUPPORTED_CITIES:
            if city in msg_lower:
                return city
        return None


# ── Groq AI validator (stage 3 — ambiguous messages only) ─────────────────────

_groq_model: Optional[ChatGroq] = None


def _get_groq_model() -> Optional[ChatGroq]:
    """
    Returns a cached Groq client for validation.
    llama-3.1-8b-instant: ~200 ms, 14,400 req/day free tier.
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


def ai_validate(message: str) -> Optional[ValidationResult]:
    """
    Validates a user message using Groq LLM (stage 3 — ambiguous messages only).

    Returns ValidationResult on success, None if Groq is unavailable or timed out
    (caller falls back to regex approval).
    """
    model = _get_groq_model()
    if model is None:
        logger.warning("AI validator: GROQ_API_KEY not set — skipping LLM stage.")
        return None

    try:
        response = model.invoke([
            SystemMessage(content=get_prompt("validator_prompt")),
            HumanMessage(content=f"Validate this user message:\n\n\"{message}\""),
        ])

        raw = response.content.strip()

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


# ── Public API ────────────────────────────────────────────────────────────────

def validate_input(user_message: str, *, is_hitl: bool = False) -> ValidationResult:
    """
    Validates one user message.

    When is_hitl=True (planner awaiting clarification), the pipeline is narrowed:
      - Short factual answers (airport codes, nationalities, durations, budgets)
        are fast-approved before any regex scan.
      - Harm and injection checks still run (always).
      - Off-topic and city checks are skipped (irrelevant for HITL answers).

    Regular flow (is_hitl=False): full three-stage pipeline.
    """
    if _is_recall_request(user_message.lower()):
        return ValidationResult(approved=True, verdict="APPROVED", reason="Profile recall request.", rejection_message="")
    
    if is_hitl:
        # Fast-approve obvious clarification answers before any regex work.
        if _is_hitl_safe(user_message):
            logger.info("Validator: HITL safe-approved %r", user_message[:40])
            return ValidationResult(
                approved=True,
                verdict="APPROVED",
                reason="HITL clarification answer — safe travel detail.",
                rejection_message="",
            )

        # Still block explicit harm or injection even in HITL mode.
        harm = InputValidator.detect_harm(user_message)
        if harm:
            verdict, reason = harm
            logger.info("Validator: HITL harm blocked. verdict=%s", verdict)
            return ValidationResult(
                approved=False,
                verdict=verdict,
                reason=reason,
                rejection_message=_REJECTION_MESSAGES["BLOCKED_HARM"],
            )

        for pattern in InputValidator._injection_patterns():
            if pattern.search(user_message):
                logger.info("Validator: HITL injection blocked.")
                return ValidationResult(
                    approved=False,
                    verdict="BLOCKED_INJECTION",
                    reason="Prompt injection attempt detected in HITL reply.",
                    rejection_message=_REJECTION_MESSAGES["BLOCKED_INJECTION"],
                )

        return ValidationResult(
            approved=True,
            verdict="APPROVED",
            reason="HITL reply passed harm and injection checks.",
            rejection_message="",
        )

    # ── Normal (non-HITL) path ────────────────────────────────────────────────
    return InputValidator.validate(user_message)
