"""
Experience sub-agent — fetches activities, weather, restaurants, events,
and local transport for the destination city.
"""

import asyncio

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.db_tools import (
    events_finder,
    fetch_activities,
    fetch_restaurants,
    fetch_weather,
    local_transport_guide,
)
from src.utils.logger import get_logger

logger = get_logger("experience_agent")


class ExperienceAgent(BaseSubAgent):
    """
    Handles:
      - activities
      - restaurants
      - local transport guide
      - weather (when travel month is known)
      - events (when travel month is known)
    """

    agent_name = "experience_agent"
    result_keys = (
        "fetch_activities",
        "fetch_restaurants",
        "local_transport_guide",
        "fetch_weather",
        "events_finder",
    )

    async def run(
        self,
        *,
        context: TripContext,
    ) -> PlannerToolResults:
        """
        Runs all experience-related tasks in parallel and returns results.
        """
        result = PlannerToolResults()

        if not context.destination_city:
            return result

        tasks = {
            "fetch_activities": asyncio.to_thread(
                fetch_activities.invoke,
                {"city": context.destination_city},
            ),
            "fetch_restaurants": asyncio.to_thread(
                fetch_restaurants.invoke,
                {"city": context.destination_city},
            ),
            "local_transport_guide": asyncio.to_thread(
                local_transport_guide.invoke,
                {"city": context.destination_city},
            ),
        }

        if context.travel_month:
            tasks["fetch_weather"] = asyncio.to_thread(
                fetch_weather.invoke,
                {"city": context.destination_city, "month": context.travel_month},
            )
            tasks["events_finder"] = asyncio.to_thread(
                events_finder.invoke,
                {"city": context.destination_city, "month": context.travel_month},
            )

        keys = list(tasks.keys())
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        for key, value in zip(keys, results):
            if isinstance(value, Exception):
                logger.error("ExperienceAgent task failed. key=%s error=%s", key, value)
                continue
            result.raw_results[key] = value

        logger.info("Experience agent fetched: %s", list(result.raw_results.keys()))

        return result
