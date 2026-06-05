"""
Central Tools Registry Module.

Exposes a unified ecosystem of local database, mathematical computation, 
and defensive external web API tools. This registry acts as the single source of truth 
for binding operational capabilities to any ReAct loop or LangGraph node execution state.

Architecture:
    - Database Tools: Factual, fixed-matrix data from the local SQLite layer.
    - Computational Tools: Deterministic, pure arithmetic calculation structures.
    - Web API Tools: Secure, real-time live network engines with armored fallbacks.
"""

from src.utils.logger import get_logger

# Import local database-driven tools
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

# Import pure arithmetic computational tools
from src.tools.calc_tools import (
    calculate_trip_cost,
    currency_conversion,
    distance_travel_time,
    estimate_daily_budget,
    generate_daily_itinerary,
    generate_trip_packages,
    summarize_trip,
)

# Import foundational legacy search tools
from src.tools.search_tools import web_search

# Import the newly engineered secure Web & Multi-API integration tools
from src.tools.web_api_tools import (
    fetch_country_metadata,
    fetch_live_events,
    fetch_live_flights,
    fetch_local_breweries,
    fetch_local_transport_live,
    geocode_location,
    live_currency_conversion,
    web_research_tavily,
)

# Planner-only tools (PDF export, final deliverables)
from src.tools.planner_tools import export_plan_to_pdf

logger = get_logger("tools_registry")

# ─── LOGICAL TOOL SEGREGATION LAYERS ────────────────────────────────────────

DATABASE_DRIVEN_TOOLS = [
    fetch_flights,
    fetch_hotels,
    fetch_activities,
    get_visa_requirement,
    list_destinations,
    get_cheapest_hotel,
    get_cheapest_flight,
    fetch_weather,
    fetch_restaurants,
    kosher_food_finder,
    events_finder,
    local_transport_guide,
    airport_transfer_info,
]

COMPUTATIONAL_CORE_TOOLS = [
    calculate_trip_cost,
    currency_conversion,
    estimate_daily_budget,
    summarize_trip,
    distance_travel_time,
    generate_daily_itinerary,
    generate_trip_packages,
]

EXTERNAL_WEB_API_TOOLS = [
    geocode_location,
    fetch_live_events,
    fetch_live_flights,
    fetch_local_transport_live,
    live_currency_conversion,
    fetch_local_breweries,
    fetch_country_metadata,
    web_research_tavily,
    web_search,
]

# Only exposed to the master planner — not the orchestrator or researcher.
PLANNER_ONLY_TOOLS = [
    export_plan_to_pdf,
]

# ─── CONSOLIDATED MASTER TOOL PORTFOLIO ─────────────────────────────────────

ALL_TOOLS = DATABASE_DRIVEN_TOOLS + COMPUTATIONAL_CORE_TOOLS + EXTERNAL_WEB_API_TOOLS

logger.info(
    "Tool registry ready. total=%d db=%d calc=%d web=%d planner_only=%d",
    len(ALL_TOOLS),
    len(DATABASE_DRIVEN_TOOLS),
    len(COMPUTATIONAL_CORE_TOOLS),
    len(EXTERNAL_WEB_API_TOOLS),
    len(PLANNER_ONLY_TOOLS),
)

# ─── EXPLICIT EXPORT CONFIGURATION ──────────────────────────────────────────

__all__ = [
    "ALL_TOOLS",
    "DATABASE_DRIVEN_TOOLS",
    "COMPUTATIONAL_CORE_TOOLS",
    "EXTERNAL_WEB_API_TOOLS",
    "PLANNER_ONLY_TOOLS",
]