import asyncio

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerToolResults, VisaResult
from src.models.trip_context import TripContext
from src.tools.db_tools import (
    fetch_flights,
    get_visa_requirement,
)
from src.utils.logger import get_logger

logger = get_logger("transport_agent")


class TransportAgent(BaseSubAgent):
    """
    Handles:
      - flights
      - visa requirements
    """

    agent_name = "transport_agent"
    result_keys = ("fetch_flights", "check_visa")

    async def run(
        self,
        *,
        context: TripContext,
    ) -> PlannerToolResults:
        """
        Runs transport-related tasks and returns independent results.
        """
        result = PlannerToolResults()

        await asyncio.gather(
            self._handle_flights(
                context=context,
                result=result,
            ),
            self._handle_visa(
                context=context,
                result=result,
            ),
        )

        return result

    async def _handle_flights(
        self,
        *,
        context: TripContext,
        result: PlannerToolResults,
    ) -> None:
        if not context.origin_airport or not context.destination_city:
            return

        raw = await asyncio.to_thread(
            fetch_flights.invoke,
            {
                "origin": context.origin_airport,
                "destination": context.destination_city,
            },
        )

        result.raw_results["fetch_flights"] = raw

        logger.info("Transport agent fetched flights.")

    async def _handle_visa(
        self,
        *,
        context: TripContext,
        result: PlannerToolResults,
    ) -> None:
        if not context.origin_country or not context.destination_country:
            return

        raw = await asyncio.to_thread(
            get_visa_requirement.invoke,
            {
                "origin_country": context.origin_country,
                "destination_country": context.destination_country,
            },
        )

        result.raw_results["check_visa"] = raw

        result.visa = VisaResult(
            origin_country=context.origin_country,
            destination_country=context.destination_country,
            requirement_summary=str(raw),
        )

        logger.info("Transport agent checked visa.")