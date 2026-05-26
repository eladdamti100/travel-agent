"""
Session models — session ID validation before use as a LangGraph thread_id.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionValidationResult:
    """
    Result returned after validating a session ID.
    """

    is_valid: bool
    error_message: str = ""


_SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,50}$")

_BANNED_SESSION_WORDS = frozenset({
    # Violence / harm
    "kill", "murder", "bomb", "attack", "shoot", "stab", "harm", "hurt",

    # Hacking / illegal
    "hack", "crack", "exploit", "inject", "exec", "eval",

    # SQL / system abuse
    "drop", "delete", "truncate", "insert", "select", "update",

    # Injection keywords
    "ignore", "override", "bypass", "jailbreak", "dan",
    "forget", "disregard", "prompt", "system",
})


def validate_session_id(session_id: str) -> SessionValidationResult:
    """
    Validates the session ID before it is used as a LangGraph thread_id.

    Rules:
      1. The session ID must not be empty.
      2. The session ID may only contain letters, numbers, underscores, and hyphens.
      3. The session ID must be at most 50 characters.
      4. The session ID must not contain prohibited security-sensitive words.
    """
    if not session_id or not session_id.strip():
        return SessionValidationResult(
            is_valid=False,
            error_message="Session ID cannot be empty.",
        )

    if not _SESSION_ID_PATTERN.match(session_id):
        return SessionValidationResult(
            is_valid=False,
            error_message=(
                "Session ID may only contain letters, numbers, underscores (_) and hyphens (-). "
                "Maximum 50 characters."
            ),
        )

    words = set(re.split(r"[_\-]", session_id.lower()))
    banned = words & _BANNED_SESSION_WORDS

    if banned:
        return SessionValidationResult(
            is_valid=False,
            error_message=f"Session ID contains a prohibited word: '{next(iter(banned))}'.",
        )

    return SessionValidationResult(is_valid=True)