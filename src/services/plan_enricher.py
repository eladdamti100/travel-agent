"""
Plan enricher — web fallback and cost calculation for the master planner.

Extracted from planner.py to keep the orchestration module focused.
"""

import asyncio
import re
from typing import Dict, Optional

from src.models.planner import PlannerTaskType
from src.models.trip_context import TripContext
from src.services.planner_result_parser import extract_lowest_price_from_json
from src.tools.calc_tools import calculate_trip_cost
from src.utils.logger import get_logger

logger = get_logger("plan_enricher")


async def fill_missing_with_web(
    context: TripContext,
    task_results: Dict[str, str],
) -> Dict[str, str]:
    """
    For each DB section that returned no data, fires a targeted Tavily web search
    and stores the result under the same task key so the final plan can surface it.

    Runs all fallback queries concurrently. Failures are logged and silently skipped.
    """
    from src.tools.web_api_tools import web_research_tavily

    city = context.destination_city or ""
    origin = context.origin_airport or ""

    def _is_empty(key: str) -> bool:
        val = task_results.get(key, "")
        if not val:
            return True
        return val.strip().lower().startswith("no ")

    queries: Dict[str, str] = {}

    if _is_empty(PlannerTaskType.FETCH_FLIGHTS.value) and origin and city:
        queries[PlannerTaskType.FETCH_FLIGHTS.value] = (
            f"best flights from {origin} to {city} airlines prices schedule"
        )
    if _is_empty(PlannerTaskType.FETCH_HOTELS.value) and city:
        queries[PlannerTaskType.FETCH_HOTELS.value] = (
            f"best hotels to stay in {city} price per night budget"
        )
    if _is_empty(PlannerTaskType.FETCH_ACTIVITIES.value) and city:
        queries[PlannerTaskType.FETCH_ACTIVITIES.value] = (
            f"top tourist attractions and activities in {city} with prices"
        )
    if _is_empty(PlannerTaskType.CHECK_VISA.value) and context.origin_country and city:
        queries[PlannerTaskType.CHECK_VISA.value] = (
            f"visa requirements for {context.origin_country} passport to visit {city}"
        )

    if not queries:
        return task_results

    logger.info("Web fallback triggered for missing DB keys: %s", list(queries.keys()))

    async def _fetch(key: str, query: str) -> tuple:
        try:
            result = await web_research_tavily.ainvoke({"query": query})
            return key, str(result)
        except Exception as exc:
            logger.warning("Web fallback failed for key=%s: %s", key, exc)
            return key, ""

    pairs = await asyncio.gather(*[_fetch(k, q) for k, q in queries.items()])
    updated = dict(task_results)
    for key, result in pairs:
        if result and not result.startswith("Search Engine"):
            updated[key] = f"[Web source] {result}"
            logger.info("Web fallback stored result for key=%s (%d chars)", key, len(result))

    return updated


async def calculate_cost_if_possible(
    context: TripContext,
    task_results: Dict[str, str],
) -> Optional[str]:
    """
    Calculates total trip cost when flight, hotel, and duration are available.
    """
    if context.duration_days is None:
        return None

    flight_price = extract_lowest_price_from_json(
        task_results.get(PlannerTaskType.FETCH_FLIGHTS.value, "")
    )
    hotel_price = extract_lowest_price_from_json(
        task_results.get(PlannerTaskType.FETCH_HOTELS.value, ""),
        price_key="price_per_night",
    )

    if hotel_price is None:
        logger.info(
            "Cost calculation skipped: no hotel data. flight_price=%s duration_days=%s",
            flight_price,
            context.duration_days,
        )
        return None

    if not flight_price:
        # DB returned no flight price; try to extract an estimate from the Tier 2
        # web agent's transport_live_research text (e.g. "$450" or "$1,200").
        web_transport = task_results.get("transport_live_research", "")
        if web_transport:
            price_match = re.search(r'\$\s*(\d[\d,]*(?:\.\d+)?)', web_transport)
            if price_match:
                try:
                    flight_price = float(price_match.group(1).replace(",", ""))
                    logger.info(
                        "Cost calculation: extracted flight price from web data. price=%.2f",
                        flight_price,
                    )
                except ValueError:
                    pass
        if not flight_price:
            logger.info("Cost calculation: no flight data — using 0 as flight cost placeholder.")
            flight_price = 0.0

    logger.info(
        "Calculating trip cost. flight_price=%s hotel_price=%s duration_days=%s",
        flight_price,
        hotel_price,
        context.duration_days,
    )

    return await asyncio.to_thread(
        calculate_trip_cost.invoke,
        {
            "flight_price": flight_price,
            "hotel_price_per_night": hotel_price,
            "duration_days": context.duration_days,
        },
    )
