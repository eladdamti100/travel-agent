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

import asyncio
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Pattern, Tuple

import httpx

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


# ── Presidio PII redaction (lazy-initialized on first use) ───────────────────
# Engines are heavy (loads spaCy model) so they are created once and cached.
# _presidio_available = None  → not yet checked
# _presidio_available = True  → engines ready
# _presidio_available = False → unavailable (spaCy model not installed)

_presidio_lock = threading.Lock()
_presidio_analyzer: Optional[Any] = None
_presidio_anonymizer: Optional[Any] = None
_presidio_available: Optional[bool] = None


def _get_presidio_engines() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (AnalyzerEngine, AnonymizerEngine) or (None, None) if unavailable."""
    global _presidio_analyzer, _presidio_anonymizer, _presidio_available

    if _presidio_available is not None:
        return _presidio_analyzer, _presidio_anonymizer

    with _presidio_lock:
        if _presidio_available is not None:
            return _presidio_analyzer, _presidio_anonymizer
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import]

            # Try each bundled spaCy model from largest to smallest so the best
            # available model is used; falls back gracefully if none are installed.
            _model: Optional[Any] = None
            for _model_name in ("en_core_web_lg", "en_core_web_md", "en_core_web_sm"):
                try:
                    from presidio_analyzer.nlp_engine import NlpEngineProvider  # type: ignore[import]
                    _provider = NlpEngineProvider(nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": _model_name}],
                    })
                    _model = _provider.create_engine()
                    break
                except Exception:
                    continue

            _presidio_analyzer = (
                AnalyzerEngine(nlp_engine=_model) if _model else AnalyzerEngine()
            )
            _presidio_anonymizer = AnonymizerEngine()
            _presidio_available = True
            logger.info("cyber_agent.presidio. status=initialized")
        except Exception as exc:
            _presidio_available = False
            logger.warning(
                "cyber_agent.presidio. status=unavailable "
                "hint='run: python -m spacy download en_core_web_sm' error=%s",
                exc,
            )

    return _presidio_analyzer, _presidio_anonymizer


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

    # ── Optional async external security APIs (all fail-safe) ────────────────

    async def check_prompt_injection(self, text: str) -> bool:
        """
        Returns True when a prompt-injection attempt is detected.

        Primary path: Lakera Guard /v2/guard endpoint.
        Fallback: existing compiled regex patterns (always available).

        The regex fallback ensures this method never raises — the caller
        can treat a True result as a hard block.
        """
        if not text:
            return False

        from src.config.settings import settings  # local import — avoids circular

        api_key: Optional[str] = settings.lakera_api_key
        if not api_key:
            return any(p.search(text) for p in _get_injection_patterns())

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(
                    "https://api.lakera.ai/v2/guard",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"messages": [{"role": "user", "content": text[:2000]}]},
                )
            if response.status_code == 200:
                flagged: bool = (
                    response.json().get("results", [{}])[0].get("flagged", False)
                )
                if flagged:
                    logger.warning(
                        "cyber_agent.lakera. status=injection_detected text_len=%d",
                        len(text),
                    )
                return flagged
            logger.warning(
                "cyber_agent.lakera. status=http_error code=%d using=regex_fallback",
                response.status_code,
            )
        except Exception as exc:
            logger.error(
                "cyber_agent.lakera. status=exception error=%s using=regex_fallback", exc
            )

        return any(p.search(text) for p in _get_injection_patterns())

    async def check_urls(self, urls: List[str]) -> List[str]:
        """
        Returns the subset of *urls* flagged as malicious.

        Primary path: Google Safe Browsing v4 threatMatches:find.
        Fallback: empty list (fail-open — do not block legitimate content
        just because the key is missing or the API is unreachable).
        """
        if not urls:
            return []

        from src.config.settings import settings

        api_key: Optional[str] = settings.google_safe_browsing_key
        if not api_key:
            return []

        threat_entries = [{"url": u} for u in urls[:500]]
        payload = {
            "client": {"clientId": "marco-travel-planner", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": threat_entries,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(
                    "https://safebrowsing.googleapis.com/v4/threatMatches:find",
                    params={"key": api_key},
                    json=payload,
                )
            if response.status_code == 200:
                matches = response.json().get("matches", [])
                malicious = [m["threat"]["url"] for m in matches if "threat" in m]
                if malicious:
                    logger.warning(
                        "cyber_agent.safe_browsing. status=malicious_urls_found count=%d",
                        len(malicious),
                    )
                return malicious
            logger.warning(
                "cyber_agent.safe_browsing. status=http_error code=%d returning=empty",
                response.status_code,
            )
        except Exception as exc:
            logger.error(
                "cyber_agent.safe_browsing. status=exception error=%s returning=empty", exc
            )

        return []

    async def redact_sensitive_data(self, text: str) -> str:
        """
        Returns *text* with PII masked (emails, phones, credit cards, names, etc.)
        using Microsoft Presidio — fully local, no API key required.

        Fallback: original text unmodified when Presidio is unavailable (e.g. the
        spaCy model has not been downloaded yet).  Never crashes the planner.

        To enable: python -m spacy download en_core_web_sm
        """
        if not text:
            return text

        analyzer, anonymizer = _get_presidio_engines()
        if analyzer is None:
            return text

        try:
            results = await asyncio.to_thread(
                analyzer.analyze,
                text=text[:10_000],
                language="en",
            )
            if not results:
                return text
            anonymized = await asyncio.to_thread(
                anonymizer.anonymize,
                text=text[:10_000],
                analyzer_results=results,
            )
            if results:
                logger.info(
                    "cyber_agent.presidio. status=redacted entities_found=%d",
                    len(results),
                )
            return anonymized.text
        except Exception as exc:
            logger.error(
                "cyber_agent.presidio. status=exception error=%s returning=original",
                exc,
            )
            return text
