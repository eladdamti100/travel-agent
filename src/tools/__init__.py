"""
Central tool registry — import ALL_TOOLS from here to bind to any agent.
"""

from src.tools.calc_tools import (
    calculate_trip_cost,
    currency_conversion,
    distance_travel_time,
    estimate_daily_budget,
    summarize_trip,
)
from src.tools.db_tools import (
    airport_transfer_info,
    events_finder,
    fetch_activities,
    fetch_flights,
    fetch_hotels,
    fetch_restaurants,
    fetch_weather,
    get_cheapest_flight,
    get_cheapest_hotel,
    get_visa_requirement,
    kosher_food_finder,
    list_destinations,
    local_transport_guide,
)
from src.tools.search_tools import web_search

ALL_TOOLS = [
    fetch_flights,
    fetch_hotels,
    fetch_activities,
    get_visa_requirement,
    list_destinations,
    get_cheapest_hotel,
    get_cheapest_flight,
    calculate_trip_cost,
    currency_conversion,
    estimate_daily_budget,
    summarize_trip,
    fetch_weather,
    fetch_restaurants,
    kosher_food_finder,
    events_finder,
    local_transport_guide,
    airport_transfer_info,
    distance_travel_time,
    web_search,
]

__all__ = ["ALL_TOOLS"] + [t.name for t in ALL_TOOLS]
