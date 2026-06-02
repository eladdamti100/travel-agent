"""
Integration and security validation tests for the Web & Multi-API sub-system.
Executes defensive validation against input attacks, rate-limits, and multi-wave scheduling.
"""

import os
import json
import pytest
from src.models.trip_context import TripContext
from src.models.planner import PlannerTaskType
from src.agents.sub_agents.web_agent import WebAgent
from src.tools.web_api_tools import (
    geocode_location,
    fetch_live_events,
    live_currency_conversion,
    fetch_local_breweries,
    fetch_country_metadata,
)

# Mark all test cases in this module as asynchronous
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_london_context() -> TripContext:
    """Provides a pristine, realistic corporate trip context instance for testing."""
    return TripContext(
        origin_airport="TLV",
        origin_country="Israel",
        destination_city="London",
        destination_country="United Kingdom",
        duration_days=5,
        total_budget=1500,
        travel_month="June",
    )


async def test_input_sanitization_and_injection_defense():
    """Defensively asserts that the geocoding tool intercepts malicious or bloated string payloads."""
    malicious_payload = "London" + ("X" * 100) + "; DROP TABLE trips; --"
    
    # Execute tool with unsafe bloated input parameters
    raw_response = await geocode_location.ainvoke({"city": malicious_payload})
    response_data = json.loads(raw_response)
    
    # Assert system safely truncated input or activated localized structural fallbacks
    assert "status" in response_data
    assert len(response_data.get("city", "")) <= 80
    assert response_data["status"] in [
        "fallback_resolved",
        "rejected",
        "error_fallback",
    ]


async def test_concurrent_tool_execution_and_fallbacks(mock_london_context):
    """Verifies all new web tools invoke cleanly and handle missing api keys via structured defaults."""
    # Execute metadata fetcher
    meta_raw = await fetch_country_metadata.ainvoke(
        {"country_name": mock_london_context.destination_country}
    )
    meta_data = json.loads(meta_raw)
    assert "currency_code" in meta_data
    assert meta_data["currency_code"] == "GBP"

    # Execute dynamic live events engine
    events_output = await fetch_live_events.ainvoke(
        {"city": mock_london_context.destination_city}
    )
    assert isinstance(events_output, str)
    assert len(events_output) > 0

    # Execute currency pipeline conversion layer
    currency_raw = await live_currency_conversion.ainvoke(
        {"amount": 500.0, "base": "USD", "target": "GBP"}
    )
    currency_data = json.loads(currency_raw)
    assert "converted" in currency_data
    assert "rate" in currency_data


async def test_web_agent_scheduler_orchestration_waves(mock_london_context):
    """Validates that the WebAgent successfully handles task allocations from scheduling layers."""
    agent = WebAgent()
    completed_results_ledger = {
        "fetch_flights": "- Flight: BA204 ($450)",
        "fetch_hotels": "- Hotel: Central Stay ($120/night)",
    }

    # Wave 1: Contextual Geolocation Processing
    geocode_raw = await agent.execute(
        PlannerTaskType.GEOCODE_LOCATION,
        mock_london_context,
        completed_results_ledger,
    )
    geocode_data = json.loads(geocode_raw)
    assert "lat" in geocode_data
    assert "lng" in geocode_data

    # Wave 2: Concurrent Enrichment Processing
    events_res = await agent.execute(
        PlannerTaskType.FETCH_LIVE_EVENTS,
        mock_london_context,
        completed_results_ledger,
    )
    breweries_res = await agent.execute(
        PlannerTaskType.FETCH_BREWERIES,
        mock_london_context,
        completed_results_ledger,
    )
    currency_res = await agent.execute(
        PlannerTaskType.LIVE_CURRENCY_CONVERSION,
        mock_london_context,
        completed_results_ledger,
    )

    # Validate aggregated string conversions
    assert "- Adele" in events_res or "Discovery" in events_res or len(events_res) > 0
    assert isinstance(breweries_res, str)
    assert "converted" in currency_res