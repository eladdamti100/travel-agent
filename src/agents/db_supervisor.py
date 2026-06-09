"""
DB Supervisor — pure parallel dispatcher for Tier 1 DB-backed sub-agents.

Responsibilities
----------------
  - Route TransportAgent, StayAgent, and ExperienceAgent concurrently via asyncio.gather
  - Respect existing_results to skip fully-covered agents during replanning
  - Respect allowed_tasks to limit which agents run (partial replanning)
  - Return only newly fetched results (caller merges with existing_results)

Security note
-------------
DBSupervisor has NO CyberAgent. All six Zero-Trust steps are enforced exclusively by
WebSupervisor.dispatch() before and after this dispatcher is called. DB agents are
NOT bypassed — they run inside the CyberAgent-wrapped perimeter.

CyberAgent.record_outcome() is supported via the optional `cyber_agent` parameter so
WebSupervisor can pass its instance for per-agent outcome auditing without coupling
DBSupervisor to the security layer.
"""

import asyncio
from typing import Any, Dict, Optional, Set

from src.agents.dispatch_tracker import mark_agent_done, mark_agent_started
from src.agents.task_registry import get_db_agents, get_db_agents_for_tasks
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("db_supervisor")


class DBSupervisor:
    """
    Tier 1 DB agent dispatcher.

    Runs TransportAgent, StayAgent, and ExperienceAgent in parallel.
    Has no security boundary — CyberAgent wrapping is enforced by WebSupervisor.
    """

    agent_name = "db_supervisor"

    async def dispatch(
        self,
        *,
        context: TripContext,
        existing_results: Optional[Dict[str, str]] = None,
        allowed_tasks: Optional[Set[Any]] = None,
        cyber_agent: Optional[Any] = None,
    ) -> Dict[str, str]:
        """
        Selects and runs Tier 1 DB sub-agents in parallel.

        Returns only new results. existing_results is used for agent-skip
        coverage checks but is NOT included in the return value — the calling
        WebSupervisor seeds merged_raw_results with it before calling this method.

        Args:
            context:          Vetted (sanitized) TripContext from WebSupervisor.
            existing_results: Already-computed results; agents fully covered are skipped.
            allowed_tasks:    If set, only agents covering at least one task are selected.
            cyber_agent:      Optional CyberAgent ref for record_outcome auditing.
        """
        covered: Set[str] = set(existing_results or {})

        candidates = (
            get_db_agents_for_tasks(allowed_tasks)
            if allowed_tasks is not None
            else get_db_agents()
        )
        agents = [
            agent for agent in candidates
            if not all(key in covered for key in agent.result_keys)
        ]

        logger.info(
            "DBSupervisor routing. selected_agents=%s allowed_tasks=%s",
            [getattr(a, "agent_name", a.__class__.__name__) for a in agents],
            sorted(allowed_tasks) if allowed_tasks is not None else None,
        )

        if not agents:
            logger.info("DBSupervisor: all DB results already covered, skipping dispatch.")
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
                    "DBSupervisor: agent failed. agent=%s error_type=%s error=%s",
                    agent_name, type(result).__name__, result,
                )
                if cyber_agent is not None:
                    cyber_agent.record_outcome(agent_name, success=False)
                continue

            if cyber_agent is not None:
                cyber_agent.record_outcome(agent_name, success=True)

            logger.info(
                "DBSupervisor: agent completed. agent=%s result_keys=%s",
                agent_name, list(result.raw_results.keys()),
            )
            new_results.update(result.raw_results)

        return new_results
