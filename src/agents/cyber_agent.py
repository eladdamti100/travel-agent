"""
Cyber Agent — security gatekeeper between the network and the system.

Owned by WebSupervisor. Sits at the network boundary and:
  1. Sanitizes outbound text (TripContext fields headed to external APIs/tools)
     to strip prompt-injection attempts before they leave the system.
  2. Inspects inbound web-tool results for malicious-content signatures
     (script injection, SQL/shell injection, etc.) before the planner trusts them.
  3. Tracks per-service failure streaks and opens a circuit breaker so the
     supervisor/agents can skip live calls and trust static fallback data
     when an external API is failing or rate-limiting (DoS resilience).

Deliberately code/regex-based — no LLM call — so the hot web-fetch path stays
fast. Mirrors the validator's "fastest path first" philosophy; an LLM
escalation stage could be bolted on later the same way validator.py escalates
ambiguous cases to Groq.
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple

from src.agents.validator_patterns import INJECTION_PATTERNS_RAW
from src.utils.logger import get_logger

logger = get_logger("cyber_agent")

# ── Circuit breaker tuning ────────────────────────────────────────────────────
_FAILURE_THRESHOLD = 3
_COOLDOWN_SECONDS = 120.0

# ── Malicious inbound-content signatures ─────────────────────────────────────
# Code/markup/command injection patterns that should never legitimately appear
# in travel data returned by a web API. Matches are redacted, not trusted.
_MALICIOUS_CONTENT_PATTERNS_RAW: List[str] = [
    r"<script[\s>]",
    r"</script>",
    r"<iframe[\s>]",
    r"javascript:",
    r"\bon\w+\s*=\s*['\"]",          # onerror="...", onclick='...'
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bos\.system\s*\(",
    r"\bsubprocess\.",
    r";\s*rm\s+-rf\b",
    r"\bunion\s+select\b",
    r"\bdrop\s+table\b",
    r"<\?php",
    r"\bcurl\s+http",
    r"\bwget\s+http",
]

# ── Lazily-compiled, cached regex sets (mirrors vector_guard's seed cache) ───
_pattern_lock = threading.Lock()
_injection_compiled: List[Pattern] = []
_malicious_compiled: List[Pattern] = []


def _get_injection_patterns() -> List[Pattern]:
    global _injection_compiled
    if _injection_compiled:
        return _injection_compiled
    with _pattern_lock:
        if not _injection_compiled:
            _injection_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS_RAW]
    return _injection_compiled


def _get_malicious_content_patterns() -> List[Pattern]:
    global _malicious_compiled
    if _malicious_compiled:
        return _malicious_compiled
    with _pattern_lock:
        if not _malicious_compiled:
            _malicious_compiled = [re.compile(p, re.IGNORECASE) for p in _MALICIOUS_CONTENT_PATTERNS_RAW]
    return _malicious_compiled


# ── Circuit-breaker state (module-level, shared across CyberAgent instances) ─

@dataclass
class _ServiceState:
    consecutive_failures: int = 0
    opened_at: float = field(default=0.0)


_service_state: Dict[str, _ServiceState] = {}
_state_lock = threading.Lock()


class CyberAgent:
    """Gatekeeper between the network and the system — owned by WebSupervisor."""

    agent_name = "cyber_agent"

    # ── Outbound: prompt-injection sanitization ──────────────────────────────

    def sanitize_outbound(self, value: str, *, field_name: str = "value") -> str:
        """
        Strips prompt-injection phrases from text before it is sent to an
        external API/tool. Redacts matches and logs a warning rather than
        raising — a single suspicious field should never abort a planning run.
        """
        if not value:
            return value

        cleaned = value
        for pattern in _get_injection_patterns():
            if pattern.search(cleaned):
                logger.warning(
                    "cyber_agent.sanitize_outbound. field=%s action=redact pattern=%r",
                    field_name, pattern.pattern[:60],
                )
                cleaned = pattern.sub(" ", cleaned)

        return cleaned.strip()

    # ── Inbound: malicious-content inspection ────────────────────────────────

    def inspect_inbound(self, raw_results: Dict[str, str]) -> Dict[str, str]:
        """
        Scans each raw web-tool result string for malicious-content signatures
        (script/markup/command/SQL injection). Matches are redacted in place
        and logged; clean strings pass through untouched.
        """
        inspected: Dict[str, str] = {}

        for key, value in raw_results.items():
            if not isinstance(value, str) or not value:
                inspected[key] = value
                continue

            cleaned = value
            for pattern in _get_malicious_content_patterns():
                if pattern.search(cleaned):
                    logger.warning(
                        "cyber_agent.inspect_inbound. key=%s action=redact pattern=%r",
                        key, pattern.pattern[:60],
                    )
                    cleaned = pattern.sub("[redacted]", cleaned)

            inspected[key] = cleaned

        return inspected

    # ── Circuit breaker / DoS fallback management ────────────────────────────

    def record_outcome(self, service: str, *, success: bool) -> None:
        """Feeds the per-service circuit breaker with a call outcome."""
        with _state_lock:
            state = _service_state.setdefault(service, _ServiceState())

            if success:
                if state.consecutive_failures or state.opened_at:
                    logger.info("cyber_agent.circuit. service=%s status=recovered", service)
                state.consecutive_failures = 0
                state.opened_at = 0.0
                return

            state.consecutive_failures += 1
            if state.consecutive_failures >= _FAILURE_THRESHOLD and not state.opened_at:
                state.opened_at = time.monotonic()
                logger.warning(
                    "cyber_agent.circuit. service=%s status=opened consecutive_failures=%d",
                    service, state.consecutive_failures,
                )

    def is_degraded(self, service: str) -> bool:
        """
        True when `service` has tripped the circuit breaker and is still
        within its cooldown window — signals callers to skip live calls and
        trust static fallback data instead.
        """
        with _state_lock:
            state = _service_state.get(service)
            if state is None or not state.opened_at:
                return False

            if time.monotonic() - state.opened_at >= _COOLDOWN_SECONDS:
                logger.info("cyber_agent.circuit. service=%s status=cooldown_elapsed", service)
                state.consecutive_failures = 0
                state.opened_at = 0.0
                return False

            return True
