"""
Web Supervisor — router between the Master Planner and the sub-agent teams.

Security flow (Zero-Trust boundary):

  OUTBOUND  →  CyberAgent.check_prompt_injection (Lakera / regex fallback)
                   ↓ [block if flagged]
               CyberAgent.sanitize_outbound (regex strip — always runs)
                   ↓
  DISPATCH  →  All selected sub-agents in parallel (asyncio.gather)
                   ↓
  INBOUND   →  Merge raw_results
               CyberAgent.check_urls (Google Safe Browsing) → replace malicious
               CyberAgent.redact_sensitive_data (Presidio local PII redaction)
               CyberAgent.inspect_inbound (regex malicious-content scan)
                   ↓
              Return Dict[str, str] to Master Planner
"""

import asyncio
import re
import threading
from typing import Dict, Optional, Set

from src.agents.cyber_agent import CyberAgent
from src.agents.task_registry import get_agents_for_tasks, get_planner_agents
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("web_supervisor")

# ── Thread-safe active-dispatch tracker (read by main.py status monitor) ──────
_dispatch_lock = threading.Lock()
_dispatch_active: set = set()


def get_active_dispatch_agents() -> list:
    """Returns a sorted snapshot of agent names currently running in dispatch()."""
    with _dispatch_lock:
        return sorted(_dispatch_active)


def _mark_agent_started(name: str) -> None:
    with _dispatch_lock:
        _dispatch_active.add(name)


def _mark_agent_done(name: str) -> None:
    with _dispatch_lock:
        _dispatch_active.discard(name)

# TripContext free-text fields vetted by the Cyber Agent before any sub-agent runs.
_OUTBOUND_CONTEXT_FIELDS = (
    "destination_city",
    "destination_country",
    "origin_country",
    "origin_airport",
)

# Minimal URL regex: http(s):// followed by ≥10 non-whitespace chars.
_URL_RE = re.compile(r"https?://[^\s\"'<>]{10,}")


class WebSupervisor:
    """Routes Master Planner tasks to the sub-agent teams via a Zero-Trust boundary."""

    agent_name = "web_supervisor"

    def __init__(self, cyber_agent: Optional[CyberAgent] = None) -> None:
        self._cyber = cyber_agent or CyberAgent()

    async def dispatch(
        self,
        *,
        context: TripContext,
        existing_results: Optional[Dict[str, str]] = None,
        allowed_tasks: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        """
        Selects, vets, and runs the relevant sub-agents in parallel; returns
        their merged raw_results dict (same shape the planner already expects).

        Security steps
        --------------
        1. Outbound async injection check (Lakera Guard / regex fallback)
           — hard-blocks dispatch if any TripContext field is flagged.
        2. Outbound regex sanitization (always runs, strips residual injections).
        3. Parallel sub-agent dispatch.
        4. Inbound URL check (Google Safe Browsing) — replaces malicious links.
        5. Inbound PII redaction (Presidio — local, no API key required).
        6. Inbound regex malicious-content scan (always runs).
        """
        covered: Set[str] = set(existing_results or {})

        candidates = (
            get_agents_for_tasks(allowed_tasks) if allowed_tasks is not None
            else get_planner_agents()
        )
        agents = [
            agent for agent in candidates
            if not all(key in covered for key in agent.result_keys)
        ]

        logger.info(
            "WebSupervisor routing. selected_agents=%s allowed_tasks=%s covered=%s",
            [getattr(a, "agent_name", a.__class__.__name__) for a in agents],
            sorted(allowed_tasks) if allowed_tasks is not None else None,
            sorted(covered),
        )

        merged_raw_results: Dict[str, str] = {**(existing_results or {})}

        if not agents:
            logger.info("WebSupervisor skipped all sub-agents — required results already covered.")
            return merged_raw_results

        # ── Step 1: Outbound async injection check ────────────────────────────
        for field_name in _OUTBOUND_CONTEXT_FIELDS:
            value: str = getattr(context, field_name, None) or ""
            if not value:
                continue
            try:
                flagged = await self._cyber.check_prompt_injection(value)
            except Exception as exc:
                logger.error(
                    "WebSupervisor: injection check threw. field=%s error=%s using=allow",
                    field_name, exc,
                )
                flagged = False

            if flagged:
                logger.error(
                    "WebSupervisor: outbound injection detected. field=%s action=block",
                    field_name,
                )
                return {
                    "security_alert": (
                        f"Planning blocked: outbound context field '{field_name}' "
                        "was flagged for prompt injection. Please revise your request."
                    )
                }

        # ── Step 2: Outbound regex sanitization (sync, always runs) ──────────
        vetted_context = self._sanitize_context(context)

        # ── Step 3: Parallel sub-agent dispatch ───────────────────────────────
        async def _run_tracked(agent):
            name = getattr(agent, "agent_name", agent.__class__.__name__)
            _mark_agent_started(name)
            try:
                return await agent.run(context=vetted_context)
            finally:
                _mark_agent_done(name)

        results = await asyncio.gather(
            *[_run_tracked(agent) for agent in agents],
            return_exceptions=True,
        )

        for agent, result in zip(agents, results):
            agent_name = getattr(agent, "agent_name", agent.__class__.__name__)

            if isinstance(result, Exception):
                logger.error(
                    "WebSupervisor: sub-agent failed. agent=%s error_type=%s error=%s",
                    agent_name, type(result).__name__, result,
                )
                self._cyber.record_outcome(agent_name, success=False)
                continue

            self._cyber.record_outcome(agent_name, success=True)
            logger.info(
                "WebSupervisor: sub-agent completed. agent=%s result_keys=%s",
                agent_name, list(result.raw_results.keys()),
            )
            merged_raw_results.update(result.raw_results)

        # ── Step 4: Inbound URL check (Google Safe Browsing) ─────────────────
        all_urls: list[str] = []
        for value in merged_raw_results.values():
            if isinstance(value, str):
                all_urls.extend(_URL_RE.findall(value))

        if all_urls:
            unique_urls = list(set(all_urls))
            try:
                malicious = await self._cyber.check_urls(unique_urls)
            except Exception as exc:
                logger.error(
                    "WebSupervisor: URL check threw. error=%s skipping=url_block", exc
                )
                malicious = []

            if malicious:
                malicious_set = set(malicious)
                for key in list(merged_raw_results.keys()):
                    for bad_url in malicious_set:
                        merged_raw_results[key] = merged_raw_results[key].replace(
                            bad_url, "[BLOCKED MALICIOUS URL]"
                        )
                logger.warning(
                    "WebSupervisor: malicious URLs blocked. count=%d", len(malicious_set)
                )

        # ── Step 5: Inbound PII redaction (Presidio — local) ─────────────────
        for key in list(merged_raw_results.keys()):
            try:
                merged_raw_results[key] = await self._cyber.redact_sensitive_data(
                    merged_raw_results[key]
                )
            except Exception as exc:
                logger.error(
                    "WebSupervisor: PII redaction threw. key=%s error=%s skipping=redact",
                    key, exc,
                )

        # ── Step 6: Inbound regex malicious-content scan (always runs) ───────
        return self._cyber.inspect_inbound(merged_raw_results)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sanitize_context(self, context: TripContext) -> TripContext:
        """Runs outbound free-text fields through the regex Cyber Agent sanitizer."""
        updates: Dict[str, str] = {}
        for field_name in _OUTBOUND_CONTEXT_FIELDS:
            value = getattr(context, field_name, None)
            if isinstance(value, str) and value:
                cleaned = self._cyber.sanitize_outbound(value, field_name=field_name)
                if cleaned != value:
                    updates[field_name] = cleaned

        return context.model_copy(update=updates) if updates else context
