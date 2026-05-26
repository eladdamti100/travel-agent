import asyncio

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.db_tools import fetch_hotels
from src.utils.logger import get_logger

logger = get_logger("stay_agent")


class StayAgent(BaseSubAgent):
    """
    Handles accommodation and hotels.
    """

    agent_name = "stay_agent"
    result_keys = ("fetch_hotels",)

    async def run(
        self,
        *,
        context: TripContext,
    ) -> PlannerToolResults:
        """
        Runs accommodation-related tasks and returns independent results.
        """
        result = PlannerToolResults()

        if not context.destination_city:
            return result

        raw = await asyncio.to_thread(
            fetch_hotels.invoke,
            {
                "city": context.destination_city,
            },
        )

        result.raw_results["fetch_hotels"] = raw

        logger.info("Stay agent fetched hotels.")

        return result