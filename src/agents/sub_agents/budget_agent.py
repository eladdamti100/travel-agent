import asyncio
import json
from typing import Optional

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import CostResult, PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.calc_tools import calculate_trip_cost
from src.utils.logger import get_logger

logger = get_logger("budget_agent")


def _lowest_price(raw_json: str, price_key: str = "price") -> Optional[float]:
    """Extract the lowest numeric price from a JSON tool response."""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        v = data.get(price_key)
        return float(v) if v is not None else None
    if isinstance(data, list):
        prices = []
        for item in data:
            if isinstance(item, dict):
                v = item.get(price_key)
                if v is not None:
                    try:
                        prices.append(float(v))
                    except (TypeError, ValueError):
                        pass
        return min(prices) if prices else None
    return None


class BudgetAgent(BaseSubAgent):
    """
    Handles trip cost estimation using real flight and hotel prices from shared_results.
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

        raw_results = shared_results.raw_results
        flight_price = _lowest_price(raw_results.get("fetch_flights", ""))
        hotel_price = _lowest_price(
            raw_results.get("fetch_hotels", ""), price_key="price_per_night"
        )

        if flight_price is None or hotel_price is None:
            logger.info("Budget agent skipped: flight or hotel price unavailable.")
            return shared_results

        raw = await asyncio.to_thread(
            calculate_trip_cost.invoke,
            {
                "flight_price": flight_price,
                "hotel_price_per_night": hotel_price,
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
