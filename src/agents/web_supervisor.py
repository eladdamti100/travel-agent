"""
Web Supervisor — security-aware coordinator between the Master Planner and both
Tier 1 DB and Tier 2 Web sub-agent teams.

Security flow (Zero-Trust boundary, 6 steps):

  OUTBOUND  →  Step 1: CyberAgent.check_prompt_injection (Lakera v2 / regex fallback)
                           ↓ [hard-block if flagged]
               Step 2: CyberAgent.sanitize_outbound (regex strip — always runs)
                           ↓
  DISPATCH  →  Step 3: Concurrent dual dispatch
                 ├─ DBSupervisor.dispatch()    → TransportAgent, StayAgent, ExperienceAgent
                 └─ _dispatch_web_agents()     → TransportWebAgent, StayWebAgent,
                                                  ExperienceWebAgent, ManagerWebAgent
               Both tiers run via asyncio.gather; exceptions are caught per-tier.
                           ↓
  INBOUND   →  Merge raw_results from both tiers + existing_results
               Step 4: CyberAgent.check_urls (Google Safe Browsing) → block malicious
               Step 5: CyberAgent.redact_sensitive_data (Presidio — local, no API key)
               Step 6: CyberAgent.inspect_inbound (regex malicious-content scan)
                           ↓
              Return Dict[str, str] to Master Planner

Both DB and Web agents pass through the full 6-step CyberAgent pipeline — no tier
bypasses the security boundary.
"""

import asyncio
import re
from typing import Any, Dict, Optional, Set

from src.agents.cyber_agent import CyberAgent
from src.agents.db_supervisor import DBSupervisor
from src.agents.dispatch_tracker import (
    get_active_dispatch_agents,  # re-exported — main.py imports from here
    mark_agent_done,
    mark_agent_started,
)
from src.agents.task_registry import get_web_agents, get_web_agents_for_tasks
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("web_supervisor")

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
    """
    Security-aware coordinator for all sub-agent tiers.

    Owns the CyberAgent Zero-Trust boundary and orchestrates:
      - DBSupervisor        → Tier 1 DB-backed agents (SQLite tools only)
      - _dispatch_web_agents → Tier 2 hierarchical web agents (live APIs, LLM loops)

    Both tiers run concurrently at Step 3. CyberAgent outbound checks (Steps 1-2)
    run before either tier is dispatched. Inbound checks (Steps 4-6) run on the
    merged results from both tiers before returning to the Master Planner.

    Accepts an optional `db_supervisor` parameter for dependency injection in tests.
    """

    agent_name = "web_supervisor"

    def __init__(
        self,
        cyber_agent: Optional[CyberAgent] = None,
        db_supervisor: Optional[DBSupervisor] = None,
    ) -> None:
        self._cyber = cyber_agent or CyberAgent()
        self._db_supervisor = db_supervisor or DBSupervisor()

    async def dispatch(
        self,
        *,
        context: TripContext,
        existing_results: Optional[Dict[str, str]] = None,
        allowed_tasks: Optional[Set[Any]] = None,
    ) -> Dict[str, str]:
        """
        Vets context, dispatches both tiers concurrently, inspects merged results.

        Security steps
        --------------
        1. Outbound async injection check (Lakera Guard v2 / regex fallback)
           — hard-blocks dispatch if any TripContext free-text field is flagged.
        2. Outbound regex sanitization (always runs, strips residual injections).
        3. Concurrent dual dispatch:
             DBSupervisor  → Tier 1 DB agents in parallel
             _dispatch_web_agents → Tier 2 web agents in parallel
        4. Inbound URL check (Google Safe Browsing) — replaces malicious links.
        5. Inbound PII redaction (Presidio — fully local, no API key required).
        6. Inbound regex malicious-content scan (always runs).
        """
        logger.info(
            "WebSupervisor.dispatch. allowed_tasks=%s existing=%s",
            sorted(allowed_tasks) if allowed_tasks is not None else None,
            sorted(existing_results or {}),
        )

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

        # ── Step 3: Concurrent dual dispatch ─────────────────────────────────
        # Both tiers receive the same allowed_tasks and existing_results.
        # Each tier's registry call filters to only relevant agents — if allowed_tasks
        # contains only DB task keys, the web tier returns {} and vice versa.
        db_coro = self._db_supervisor.dispatch(
            context=vetted_context,
            existing_results=existing_results,
            allowed_tasks=allowed_tasks,
            cyber_agent=self._cyber,      # enables per-agent record_outcome in DB tier
        )

        web_coro = self._dispatch_web_agents(
            context=vetted_context,
            existing_results=existing_results,
            allowed_tasks=allowed_tasks,
        )

        db_new, web_new = await asyncio.gather(db_coro, web_coro, return_exceptions=True)

        merged_raw_results: Dict[str, str] = {**(existing_results or {})}

        if isinstance(db_new, dict):
            merged_raw_results.update(db_new)
        elif isinstance(db_new, Exception):
            logger.error(
                "WebSupervisor: DBSupervisor dispatch failed. error_type=%s error=%s",
                type(db_new).__name__, db_new,
            )

        if isinstance(web_new, dict):
            merged_raw_results.update(web_new)
        elif isinstance(web_new, Exception):
            logger.error(
                "WebSupervisor: web agent dispatch failed. error_type=%s error=%s",
                type(web_new).__name__, web_new,
            )

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

    async def _dispatch_web_agents(
        self,
        *,
        context: TripContext,
        existing_results: Optional[Dict[str, str]] = None,
        allowed_tasks: Optional[Set[Any]] = None,
    ) -> Dict[str, str]:
        """
        Dispatches Tier 2 hierarchical web agents in parallel.

        Returns only new results from this tier. Security is handled by the
        calling dispatch() method — this is a pure parallel runner.
        """
        covered: Set[str] = set(existing_results or {})

        candidates = (
            get_web_agents_for_tasks(allowed_tasks)
            if allowed_tasks is not None
            else get_web_agents()
        )
        agents = [
            agent for agent in candidates
            if not all(key in covered for key in agent.result_keys)
        ]

        logger.info(
            "WebSupervisor web-tier routing. selected_agents=%s",
            [getattr(a, "agent_name", a.__class__.__name__) for a in agents],
        )

        if not agents:
            logger.info("WebSupervisor: no Tier 2 web agents needed, skipping.")
            return {}

        async def _run_tracked(agent):
            name = getattr(agent, "agent_name", agent.__class__.__name__)
            mark_agent_started(name)
            try:
                return await agent.run(context=context)
            finally:
                mark_agent_done(name)

        results = await asyncio.gather(
            *[_run_tracked(agent) for agent in agents],
            return_exceptions=True,
        )

        new_results: Dict[str, str] = {}
        for agent, result in zip(agents, results):
            agent_name = getattr(agent, "agent_name", agent.__class__.__name__)

            if isinstance(result, Exception):
                logger.error(
                    "WebSupervisor: web agent failed. agent=%s error_type=%s error=%s",
                    agent_name, type(result).__name__, result,
                )
                self._cyber.record_outcome(agent_name, success=False)
                continue

            self._cyber.record_outcome(agent_name, success=True)
            logger.info(
                "WebSupervisor: web agent completed. agent=%s result_keys=%s",
                agent_name, list(result.raw_results.keys()),
            )
            new_results.update(result.raw_results)

        return new_results

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
