"""
Experience Web Agent — live research for events and local brewing scene.

Responsibilities (per Epic 3 decomposition):
  FETCH_LIVE_EVENTS  — upcoming concerts, shows, and festivals (Ticketmaster)
  FETCH_BREWERIES    — local craft brewery listings (Open Brewery DB)
  experience_web_research — synthesized activity & culture summary

Tools:
  fetch_live_events      — Ticketmaster API (static-event fallback)
  fetch_local_breweries  — Open Brewery DB (text fallback)

max_iterations = len(tools) * 2 = 4
"""

from __future__ import annotations

from typing import Dict, Tuple

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.agents.base import get_model
from src.agents.sub_agents.base import BaseSubAgent
from src.agents.sub_agents.web_agents._message_helpers import extract_tool_results
from src.models.planner import PlannerTaskType, PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.web_api_tools import fetch_live_events, fetch_local_breweries
from src.utils.logger import get_logger

logger = get_logger("experience_web_agent")

_TOOLS = [fetch_live_events, fetch_local_breweries]
_MAX_ITERATIONS: int = len(_TOOLS) * 2   # 4
_RECURSION_LIMIT: int = _MAX_ITERATIONS + 2  # 6

# Maps tool name → raw_results key for deterministic output capture
_TOOL_TO_KEY: Dict[str, str] = {
    "fetch_live_events":       PlannerTaskType.FETCH_LIVE_EVENTS.value,
    "fetch_local_breweries":   PlannerTaskType.FETCH_BREWERIES.value,
}

_SYSTEM_PROMPT = (
    "You are a specialized experience research agent for the Marco AI Travel Planner. "
    "You have two tools. For each planning request you MUST call BOTH:\n\n"
    "  1. fetch_live_events      — find upcoming concerts, festivals, and local events\n"
    "  2. fetch_local_breweries  — list notable craft breweries and taprooms\n\n"
    "Call both tools for the destination city. "
    "Summarize the findings into a short 'What to Do' brief that highlights "
    "the best events and nightlife spots the traveler should consider booking."
)


class ExperienceWebAgent(BaseSubAgent):
    """
    Autonomous web research agent for live events and local breweries.

    Owns FETCH_LIVE_EVENTS and FETCH_BREWERIES (event/entertainment domain).
    Currency, country metadata, and general research moved to ManagerWebAgent.
    Geocoding moved to TransportWebAgent.

    A fresh langgraph ReAct graph is built per run() call — isolated scratchpad,
    no shared state across concurrent sessions.
    """

    agent_name = "experience_web_agent"
    result_keys: Tuple[str, ...] = (
        PlannerTaskType.FETCH_LIVE_EVENTS.value,   # "fetch_live_events"
        PlannerTaskType.FETCH_BREWERIES.value,      # "fetch_breweries"
        "experience_web_research",
    )

    async def run(self, *, context: TripContext) -> PlannerToolResults:
        result = PlannerToolResults()

        if not context.destination_city:
            logger.info("experience_web_agent. destination_city=None skipping=True")
            return result

        query = (
            f"Research the experience scene for {context.destination_city}:\n"
            f"  1. Call fetch_live_events(city='{context.destination_city}') "
            f"to get upcoming events and concerts.\n"
            f"  2. Call fetch_local_breweries(city='{context.destination_city}') "
            f"to find local craft breweries.\n\n"
            f"Travel month: {context.travel_month or 'flexible'}. "
            f"Summarize the best events and local spots the traveler should consider."
        )

        try:
            agent = create_react_agent(
                model=get_model(),
                tools=_TOOLS,
                prompt=_SYSTEM_PROMPT,
            )

            response = await agent.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"recursion_limit": _RECURSION_LIMIT},
            )

            messages = response.get("messages", [])

            # ── Capture individual tool results under their planner keys ──
            result.raw_results.update(extract_tool_results(messages, _TOOL_TO_KEY))

            # ── Narrative 'What to Do' summary ───────────────────────────
            final_msg = messages[-1] if messages else None
            summary: str = str(getattr(final_msg, "content", "")).strip()
            if summary:
                result.raw_results["experience_web_research"] = summary

            logger.info(
                "experience_web_agent.completed. destination=%s keys=%s",
                context.destination_city, list(result.raw_results.keys()),
            )

        except Exception as exc:
            logger.error(
                "experience_web_agent.failed. destination=%s error_type=%s error=%s",
                context.destination_city, type(exc).__name__, exc,
            )
            result.raw_results["experience_web_research"] = (
                f"Live experience research unavailable for {context.destination_city}. "
                "Static activity data from the knowledge base will be used."
            )

        return result
