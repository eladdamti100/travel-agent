import asyncio

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import (
    CostResult,
    PlannerToolResults,
)
from src.models.trip_context import TripContext
from src.tools.calc_tools import calculate_trip_cost
from src.utils.logger import get_logger

logger = get_logger("budget_agent")


class BudgetAgent(BaseSubAgent):
    """
    Handles trip cost estimation and budget reasoning.
    """

    agent_name = "budget_agent"

    async def run(
        self,
        *,
        context: TripContext,
        shared_results: PlannerToolResults,
    ) -> PlannerToolResults:

        if context.duration_days is None:
            return shared_results

        raw = await asyncio.to_thread(
            calculate_trip_cost.invoke,
            {
                "flight_price": 500,
                "hotel_price_per_night": 150,
                "duration_days": context.duration_days,
            },
        )

        shared_results.raw_results["calculate_trip_cost"] = raw

        shared_results.cost = CostResult(
            duration_days=context.duration_days,
            raw={"raw_response": str(raw)},
        )

        logger.info("Budget agent calculated trip cost.")

        return shared_results