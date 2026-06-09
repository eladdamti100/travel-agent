"""
Transport Web Agent — live research for flights, transport alternatives,
visa / entry requirements, AND destination geocoding.

Responsibilities (per Epic 3 decomposition):
  GEOCODE_LOCATION      — coordinates for the destination city
  transport_live_research — narrative: flights, ground transport, visa updates

Tools:
  tavily_transport_search  — live flight / transport / visa research via Tavily
  geocode_location         — precise coordinates via OpenCage (static fallback)

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
from src.tools.web_api_tools import geocode_location, tavily_transport_search
from src.utils.logger import get_logger

logger = get_logger("transport_web_agent")

_TOOLS = [tavily_transport_search, geocode_location]
_MAX_ITERATIONS: int = len(_TOOLS) * 2   # 4
_RECURSION_LIMIT: int = _MAX_ITERATIONS + 2  # 6

# Maps tool name → raw_results key for deterministic output capture
_TOOL_TO_KEY: Dict[str, str] = {
    "geocode_location": PlannerTaskType.GEOCODE_LOCATION.value,
}

_SYSTEM_PROMPT = (
    "You are a specialized transport research agent for the Marco AI Travel Planner. "
    "You have two tools. For each planning request you MUST call BOTH:\n\n"
    "  1. geocode_location      — get the precise GPS coordinates of the destination city\n"
    "  2. tavily_transport_search — find live flight prices, ground transport options, "
    "and current visa / entry requirements\n\n"
    "Call geocode_location first, then tavily_transport_search. "
    "Report findings concisely: coordinates, flight price ranges, airport-to-city "
    "transport options, and any critical visa or travel advisory notes."
)


class TransportWebAgent(BaseSubAgent):
    """
    Autonomous web research agent for live transport data and geocoding.

    Owns GEOCODE_LOCATION (always deterministic) and produces a narrative
    summary of live transport intelligence (transport_live_research).

    A fresh langgraph ReAct graph is built per run() call — no shared state
    between concurrent planning sessions.
    """

    agent_name = "transport_web_agent"
    result_keys: Tuple[str, ...] = (
        PlannerTaskType.GEOCODE_LOCATION.value,   # "geocode_location"
        "transport_live_research",
    )

    async def run(self, *, context: TripContext) -> PlannerToolResults:
        result = PlannerToolResults()

        if not context.destination_city:
            logger.info("transport_web_agent. destination_city=None skipping=True")
            return result

        query = (
            f"Research the destination {context.destination_city}:\n"
            f"  1. Call geocode_location(city='{context.destination_city}') "
            f"to get its GPS coordinates.\n"
            f"  2. Call tavily_transport_search with query: "
            f"'live flights transport visa from {context.origin_country or 'abroad'} "
            f"to {context.destination_city} "
            f"{context.travel_month or ''}'\n\n"
            f"Budget: {'${:,.0f}'.format(context.total_budget) if context.total_budget else 'unspecified'}. "
            f"Report coordinates, flight price ranges, airport-to-city options, "
            f"and current entry requirements."
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

            # ── Capture geocode_location result under its planner key ─────
            result.raw_results.update(extract_tool_results(messages, _TOOL_TO_KEY))

            # ── Narrative summary from final AI message ───────────────────
            final_msg = messages[-1] if messages else None
            output: str = str(getattr(final_msg, "content", "")).strip() or (
                f"No live transport data available for {context.destination_city}."
            )
            result.raw_results["transport_live_research"] = output

            logger.info(
                "transport_web_agent.completed. destination=%s keys=%s",
                context.destination_city, list(result.raw_results.keys()),
            )

        except Exception as exc:
            logger.error(
                "transport_web_agent.failed. destination=%s error_type=%s error=%s",
                context.destination_city, type(exc).__name__, exc,
            )
            result.raw_results["transport_live_research"] = (
                f"Live transport research unavailable for {context.destination_city}. "
                "Consult airline websites and your destination's official embassy site "
                "for current visa and entry requirements."
            )

        return result
