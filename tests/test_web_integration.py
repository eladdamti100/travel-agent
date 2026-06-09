"""
Integration and security validation tests for the Web & Multi-API sub-system.

NOTE (Epic 3): The monolithic WebAgent was decomposed into TransportWebAgent,
StayWebAgent, ExperienceWebAgent, and ManagerWebAgent.  The WebSupervisor now
dispatches to all four via asyncio.gather.  Tests that targeted the old
WebAgent.execute() interface have been removed; new agent-level tests live in
tests/test_p0_fixes.py and tests/test_web_api_mocked.py.
"""

import json

import pytest

from src.models.trip_context import TripContext
from src.tools.web_api_tools import (
    fetch_country_metadata,
    fetch_live_events,
    live_currency_conversion,
)

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
    from src.tools.web_api_tools import geocode_location

    malicious_payload = "London" + ("X" * 100) + "; DROP TABLE trips; --"

    raw_response = await geocode_location.ainvoke({"city": malicious_payload})
    response_data = json.loads(raw_response)

    assert "status" in response_data
    assert len(response_data.get("city", "")) <= 80
    assert response_data["status"] in [
        "fallback_resolved",
        "rejected",
        "error_fallback",
    ]


async def test_concurrent_tool_execution_and_fallbacks(mock_london_context):
    """Verifies all web tools invoke cleanly and handle missing API keys via structured defaults."""
    meta_raw = await fetch_country_metadata.ainvoke(
        {"country_name": mock_london_context.destination_country}
    )
    meta_data = json.loads(meta_raw)
    assert "currency_code" in meta_data
    assert meta_data["currency_code"] == "GBP"

    events_output = await fetch_live_events.ainvoke(
        {"city": mock_london_context.destination_city}
    )
    assert isinstance(events_output, str)
    assert len(events_output) > 0

    currency_raw = await live_currency_conversion.ainvoke(
        {"amount": 500.0, "base": "USD", "target": "GBP"}
    )
    currency_data = json.loads(currency_raw)
    assert "converted" in currency_data
    assert "rate" in currency_data
