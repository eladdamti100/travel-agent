"""
Stay Web Agent — autonomous live research for hotel reviews and pricing trends.

Uses langgraph.prebuilt.create_react_agent with an isolated in-memory graph
per run(). No checkpointer is set, so each call starts with a clean scratchpad.

max_iterations is enforced via recursion_limit in the ainvoke config:
  len(tools) * 2 + 2 = 4  (1 tool × 2 + 2 overhead steps)

Result key:  "stay_live_research"
"""

from __future__ import annotations

from typing import Tuple

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from src.agents.base import get_model
from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerToolResults
from src.tools.web_api_tools import tavily_reviews_search
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("stay_web_agent")

_TOOLS = [tavily_reviews_search]
_MAX_ITERATIONS: int = len(_TOOLS) * 2  # 2
_RECURSION_LIMIT: int = _MAX_ITERATIONS + 2  # buffer for overhead steps

_SYSTEM_PROMPT = (
    "You are a specialized accommodation research agent for the Marco AI Travel Planner. "
    "Your mission is to find live hotel and stay intelligence for a planned trip:\n"
    "  1. Current average nightly rates for the destination and travel period\n"
    "  2. Top-reviewed hotel neighbourhoods and why travellers prefer them\n"
    "  3. Recent guest review trends (cleanliness, value, location scores)\n"
    "  4. Budget vs mid-range vs luxury tier breakdowns with price ranges\n"
    "  5. Any current promotions, early-bird deals, or blackout periods to avoid\n\n"
    "Use the available search tool to gather current information. "
    "Be concrete: provide price ranges (per night), star ratings, and names "
    "of notable properties where possible. "
    "If the search tool is unavailable, say so clearly."
)


class StayWebAgent(BaseSubAgent):
    """
    Autonomous web research agent for live hotel and accommodation data.

    Builds a fresh langgraph ReAct agent per run() call — isolated in-memory
    scratchpad, no persistence across concurrent planning sessions.
    """

    agent_name = "stay_web_agent"
    result_keys: Tuple[str, ...] = ("stay_live_research",)

    async def run(self, *, context: TripContext) -> PlannerToolResults:
        result = PlannerToolResults()

        if not context.destination_city:
            logger.info("stay_web_agent. destination_city=None skipping=True")
            return result

        num_travelers = context.num_travelers or 1
        duration = context.duration_days or 0

        query = (
            f"Research current hotel prices and recent guest reviews for "
            f"{context.destination_city}. "
            f"Trip details: {num_travelers} traveler(s), "
            f"{f'{duration} nights' if duration else 'duration flexible'}. "
            f"Budget: {'${:,.0f} total'.format(context.total_budget) if context.total_budget else 'flexible'}. "
            f"Travel month: {context.travel_month or 'unspecified'}. "
            f"Provide: average nightly rate tiers (budget / mid-range / luxury), "
            f"best-reviewed neighbourhoods to stay in, and any current deals."
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
            final_msg = messages[-1] if messages else None
            output: str = (
                str(getattr(final_msg, "content", "")).strip()
                if final_msg else ""
            ) or f"No live accommodation data available for {context.destination_city}."

            result.raw_results["stay_live_research"] = output

            logger.info(
                "stay_web_agent.completed. destination=%s output_len=%d",
                context.destination_city, len(output),
            )

        except Exception as exc:
            logger.error(
                "stay_web_agent.failed. destination=%s error_type=%s error=%s",
                context.destination_city, type(exc).__name__, exc,
            )
            result.raw_results["stay_live_research"] = (
                f"Live accommodation research unavailable for {context.destination_city}. "
                "Check booking platforms (Booking.com, Hotels.com) for current rates."
            )

        return result
