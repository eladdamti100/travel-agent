"""
Experience sub-agent — fetches activities for the destination city.
"""

import asyncio

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.db_tools import fetch_activities
from src.utils.logger import get_logger

logger = get_logger("experience_agent")


class ExperienceAgent(BaseSubAgent):
    """
    Handles:
      - activities
      - lightweight itinerary reasoning
    """

    agent_name = "experience_agent"
    result_keys = ("fetch_activities",)

    async def run(
        self,
        *,
        context: TripContext,
    ) -> PlannerToolResults:
        """
        Runs experience-related tasks and returns independent results.
        """
        result = PlannerToolResults()

        if not context.destination_city:
            return result

        raw = await asyncio.to_thread(
            fetch_activities.invoke,
            {
                "city": context.destination_city,
            },
        )

        result.raw_results["fetch_activities"] = raw

        logger.info("Experience agent fetched activities.")

        return result