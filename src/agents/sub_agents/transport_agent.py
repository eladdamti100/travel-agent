"""
Transport sub-agent — fetches flights and checks visa requirements.
"""

import asyncio
import time

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerToolResults, VisaResult
from src.models.trip_context import TripContext
from src.tools.db_tools import (
    fetch_flights,
    get_visa_requirement,
)
from src.tools.calc_tools import _fetch_live_flights_sync
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
        start_time = time.perf_counter()

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

        elapsed = time.perf_counter() - start_time
        logger.info("TransportAgent parallel tasks completed in %.2fs", elapsed)

        return result

    async def _handle_flights(
        self,
        *,
        context: TripContext,
        result: PlannerToolResults,
    ) -> None:
        """
        Fetch flights — tries SerpAPI Google Flights first, falls back to DB.
        """
        if not context.origin_airport or not context.destination_city:
            return

        import json as _json
        from src.tools.calc_tools import _CITY_TO_IATA

        dest_iata = _CITY_TO_IATA.get(
            (context.destination_city or "").lower(),
            (context.destination_city or "")[:3].upper(),
        )

        # ── Try live SerpAPI prices first ─────────────────────────────────────
        live = await asyncio.to_thread(
            _fetch_live_flights_sync,
            context.origin_airport.upper(),
            dest_iata,
        )

        if live:
            # Store live flights under a separate key → routed to Section 2 (Web Data)
            normalised = [
                {
                    "airline":  f["airline"],
                    "price":    f["price"],
                    "duration": f.get("duration", ""),
                    "source":   "Google Flights (live)",
                }
                for f in live
            ]
            result.raw_results["fetch_live_flights"] = _json.dumps(normalised, indent=2)
            logger.info(
                "TransportAgent: %d live SerpAPI flights stored for %s -> %s",
                len(live), context.origin_airport, context.destination_city,
            )
            return

        # ── Fallback: DB flights ──────────────────────────────────────────────
        raw = await asyncio.to_thread(
            fetch_flights.invoke,
            {
                "origin":      context.origin_airport,
                "destination": context.destination_city,
            },
        )
        result.raw_results["fetch_flights"] = raw
        logger.info("TransportAgent: using DB flights (SerpAPI unavailable).")

    async def _handle_visa(
        self,
        *,
        context: TripContext,
        result: PlannerToolResults,
    ) -> None:
        """
        Fetch visa requirements when both countries are available.
        """
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
