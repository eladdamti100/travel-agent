"""
Validator — hardcoded security guardrail, zero LLM calls.

Checks every user message for four threat categories before Marco runs:
  0. Harmful content    — violence, illegal requests, hate speech, abuse
  1. Prompt injection   — attempts to override system instructions
  2. Out-of-scope       — requests completely unrelated to travel
  3. Unsupported city   — destination not in the database

All checks are regex / keyword based and compile once at class load time.
"""

import re
from dataclasses import dataclass
from typing import Optional

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


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """
    Result returned by the input validation layer.
    """

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


# ── Main validator class ──────────────────────────────────────────────────────

class InputValidator:
    """
    Stateless hardcoded validator.  All patterns compile once on first use.
    Call InputValidator.validate(message) → ValidationResult.
    detect_harm() is a public static method and can be called independently.
    """

    SUPPORTED_CITIES = SUPPORTED_CITIES
    KNOWN_UNSUPPORTED_CITIES = KNOWN_UNSUPPORTED_CITIES
    _HARM_PATTERNS_RAW = HARM_PATTERNS_RAW
    _INJECTION_PATTERNS_RAW = INJECTION_PATTERNS_RAW
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
        """
        Returns compiled harmful-content regex patterns.
        """
        if cls._harm_compiled is None:
            cls._harm_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._HARM_PATTERNS_RAW
            ]
        return cls._harm_compiled

    @classmethod
    def _injection_patterns(cls) -> list:
        """
        Returns compiled prompt-injection regex patterns.
        """
        if cls._injection_compiled is None:
            cls._injection_compiled = [
                re.compile(p, re.IGNORECASE) for p in cls._INJECTION_PATTERNS_RAW
            ]
        return cls._injection_compiled

    @classmethod
    def _off_topic_patterns(cls) -> list:
        """
        Returns compiled off-topic regex patterns with topic labels.
        """
        if cls._off_topic_compiled is None:
            cls._off_topic_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._OFF_TOPIC_PATTERNS_RAW
            ]
        return cls._off_topic_compiled

    @staticmethod
    def detect_harm(message: str) -> Optional[tuple]:
        """
        Public static method — scans message for harmful or violating content.

        Checks for: violence, dangerous instructions, self-harm, hacking,
        hate crimes, abusive language, fraud, and illegal activity.

        Returns (verdict, reason) tuple if harm is detected, None if clean.
        Can be called independently: InputValidator.detect_harm(message)

        When harm is detected the caller should cancel all further processing
        and return the BLOCKED_HARM rejection message to the user.
        """
        for pattern, label in InputValidator._harm_patterns():
            if pattern.search(message):
                return ("BLOCKED_HARM", f"Harmful content detected: {label}")
        return None

    @classmethod
    def validate(cls, message: str) -> ValidationResult:
        """
        Validates a user message before it reaches the graph.
        """
        msg_lower = message.lower().strip()

        # ── 0. Harm check (absolute highest priority) ─────────────────────────
        harm = cls.detect_harm(message)
        if harm:
            verdict, reason = harm
            return ValidationResult(
                approved=False,
                verdict=verdict,
                reason=reason,
                rejection_message=_REJECTION_MESSAGES["BLOCKED_HARM"],
            )

        # ── 1. Injection check ────────────────────────────────────────────────
        for pattern in cls._injection_patterns():
            if pattern.search(message):
                return ValidationResult(
                    approved=False,
                    verdict="BLOCKED_INJECTION",
                    reason="Prompt injection attempt detected.",
                    rejection_message=_REJECTION_MESSAGES["BLOCKED_INJECTION"],
                )

        # ── 2. City check ─────────────────────────────────────────────────────
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

        # ── 3. Off-topic check (only if no travel keyword present) ────────────
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

        # ── 4. Approved ───────────────────────────────────────────────────────
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

        Used by nodes.py to skip the LLM validator and fast-approve obvious requests,
        saving ~200ms per typical travel message.
        """
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in cls._STRONG_TRAVEL_SIGNALS)

    @classmethod
    def _detect_city(cls, msg_lower: str) -> Optional[str]:
        """
        Finds supported or known unsupported city mentions in a lowercased message.
        """
        for city in cls.SUPPORTED_CITIES:
            if city in msg_lower:
                return city
        # Country names that map to supported cities are never unsupported
        for country in cls._VISA_COUNTRY_NAMES:
            if country in msg_lower:
                return None
        for city in cls.KNOWN_UNSUPPORTED_CITIES:
            if city in msg_lower:
                return city
        return None


# ── Public function (called by nodes.py) ─────────────────────────────────────

def validate_input(user_message: str) -> ValidationResult:
    """
    Validates one raw user message with the default InputValidator.
    """
    return InputValidator.validate(user_message)
