"""
Web Supervisor — router between the Master Planner and the four web/data
sub-agents (TransportAgent, StayAgent, ExperienceAgent, WebAgent).

Replaces the inline dispatch that used to live directly inside
run_sub_agents_async (planner.py): selects the sub-agents capable of producing
the requested tasks via the task registry, has the Cyber Agent vet traffic at
the network boundary (sanitize outbound context fields, inspect inbound
results, track per-service health), runs the selected agents concurrently, and
merges their independent raw_results into a single dict — exactly the shape
the planner already expects.
"""

import asyncio
from typing import Dict, Optional, Set

from src.agents.cyber_agent import CyberAgent
from src.agents.task_registry import get_agents_for_tasks, get_planner_agents
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("web_supervisor")

# TripContext fields that flow as free text into outbound web/tool calls —
# vetted by the Cyber Agent before any sub-agent runs.
_OUTBOUND_CONTEXT_FIELDS = (
    "destination_city",
    "destination_country",
    "origin_country",
    "origin_airport",
)


class WebSupervisor:
    """Routes Master Planner tasks to the four web/data sub-agents in real time."""

    agent_name = "web_supervisor"

    def __init__(self, cyber_agent: Optional[CyberAgent] = None):
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
        their merged raw_results dict (same shape run_sub_agents_async returned).
        """
        covered = set(existing_results or {})

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
            [getattr(agent, "agent_name", agent.__class__.__name__) for agent in agents],
            sorted(allowed_tasks) if allowed_tasks is not None else None,
            sorted(covered),
        )

        merged_raw_results: Dict[str, str] = {**(existing_results or {})}

        if not agents:
            logger.info("WebSupervisor skipped all sub-agents — required results already covered.")
            return merged_raw_results

        vetted_context = self._sanitize_context(context)

        results = await asyncio.gather(
            *[agent.run(context=vetted_context) for agent in agents],
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

        return self._cyber.inspect_inbound(merged_raw_results)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sanitize_context(self, context: TripContext) -> TripContext:
        """Runs outbound free-text fields through the Cyber Agent before dispatch."""
        updates = {}
        for field_name in _OUTBOUND_CONTEXT_FIELDS:
            value = getattr(context, field_name, None)
            if isinstance(value, str) and value:
                cleaned = self._cyber.sanitize_outbound(value, field_name=field_name)
                if cleaned != value:
                    updates[field_name] = cleaned

        return context.model_copy(update=updates) if updates else context
