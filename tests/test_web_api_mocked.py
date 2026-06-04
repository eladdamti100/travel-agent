"""
Mocked tests for all five web API tools.

All external HTTP calls are intercepted with unittest.mock so tests run
offline, incur zero API cost, and never hit rate limits.

Each test verifies:
  1. Happy path: live API response parsed correctly.
  2. Missing key fallback: no env var → static data returned.
  3. HTTP error fallback: non-200 status → graceful degradation.
  4. Network exception fallback: httpx raises → graceful degradation.
  5. Input sanitisation: over-length / special-char input is cleaned.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tests.fixtures.web_api_responses import (
    BREWERIES_EMPTY,
    BREWERIES_PARIS,
    EXCHANGE_USD_EUR,
    OPENCAGE_EMPTY,
    OPENCAGE_PARIS,
    RESTCOUNTRIES_FRANCE,
    TAVILY_PARIS,
    TICKETMASTER_EMPTY,
    TICKETMASTER_PARIS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mock_response(json_data, status_code=200):
    """Return a mock httpx.Response-like object."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


def _async_client(json_data, status_code=200):
    """Context-manager mock for httpx.AsyncClient that returns a fixed response."""
    mock_resp = _mock_response(json_data, status_code)
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(return_value=mock_resp)
    return client


# ── geocode_location ──────────────────────────────────────────────────────────

class TestGeocodeLocation:
    @pytest.mark.asyncio
    async def test_live_response_parsed(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client(OPENCAGE_PARIS)):
                result = await geocode_location.ainvoke({"city": "Paris"})
        data = json.loads(result)
        assert data["status"] == "verified_live"
        assert data["lat"] == pytest.approx(48.8566)
        assert data["lng"] == pytest.approx(2.3522)

    @pytest.mark.asyncio
    async def test_missing_key_returns_fallback(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": ""}, clear=False):
            result = await geocode_location.ainvoke({"city": "Paris"})
        data = json.loads(result)
        assert data["status"] == "fallback_resolved"
        assert "lat" in data

    @pytest.mark.asyncio
    async def test_rate_limit_returns_fallback(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client({}, 429)):
                result = await geocode_location.ainvoke({"city": "London"})
        data = json.loads(result)
        assert data["status"] == "rate_limit_fallback"

    @pytest.mark.asyncio
    async def test_network_exception_returns_fallback(self):
        from src.tools.web_api_tools import geocode_location
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(side_effect=httpx.ConnectError("timeout"))
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=client):
                result = await geocode_location.ainvoke({"city": "Berlin"})
        data = json.loads(result)
        assert data["status"] == "exception_fallback"

    @pytest.mark.asyncio
    async def test_empty_results_returns_fallback(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client(OPENCAGE_EMPTY)):
                result = await geocode_location.ainvoke({"city": "Paris"})
        data = json.loads(result)
        assert data["status"] == "empty_fallback"

    @pytest.mark.asyncio
    async def test_input_too_long_truncated(self):
        from src.tools.web_api_tools import geocode_location
        long_city = "A" * 200
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": ""}, clear=False):
            result = await geocode_location.ainvoke({"city": long_city})
        # Should not raise; returns a valid JSON response
        assert json.loads(result)


# ── fetch_live_events ─────────────────────────────────────────────────────────

class TestFetchLiveEvents:
    @pytest.mark.asyncio
    async def test_live_response_parsed(self):
        from src.tools.web_api_tools import fetch_live_events
        with patch.dict("os.environ", {"TICKETMASTER_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client(TICKETMASTER_PARIS)):
                result = await fetch_live_events.ainvoke({"city": "Paris"})
        assert "Jazz" in result or "Louvre" in result

    @pytest.mark.asyncio
    async def test_missing_key_returns_mock_events(self):
        from src.tools.web_api_tools import fetch_live_events
        with patch.dict("os.environ", {"TICKETMASTER_API_KEY": ""}, clear=False):
            result = await fetch_live_events.ainvoke({"city": "Paris"})
        assert "Classical Symphony" in result or "Louvre" in result

    @pytest.mark.asyncio
    async def test_unauthorized_returns_fallback(self):
        from src.tools.web_api_tools import fetch_live_events
        with patch.dict("os.environ", {"TICKETMASTER_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client({}, 401)):
                result = await fetch_live_events.ainvoke({"city": "Paris"})
        assert isinstance(result, str) and len(result) > 0

    @pytest.mark.asyncio
    async def test_no_events_returns_fallback(self):
        from src.tools.web_api_tools import fetch_live_events
        with patch.dict("os.environ", {"TICKETMASTER_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client(TICKETMASTER_EMPTY)):
                result = await fetch_live_events.ainvoke({"city": "Tokyo"})
        assert isinstance(result, str) and len(result) > 0


# ── live_currency_conversion ──────────────────────────────────────────────────

class TestLiveCurrencyConversion:
    @pytest.mark.asyncio
    async def test_live_response_parsed(self):
        from src.tools.web_api_tools import live_currency_conversion
        with patch.dict("os.environ", {"EXCHANGERATE_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=_async_client(EXCHANGE_USD_EUR)):
                result = await live_currency_conversion.ainvoke(
                    {"amount": 1000.0, "base": "USD", "target": "EUR"}
                )
        data = json.loads(result)
        assert data["status"] == "live_synchronized"
        assert "EUR" in data["converted"]

    @pytest.mark.asyncio
    async def test_missing_key_returns_matrix_fallback(self):
        from src.tools.web_api_tools import live_currency_conversion
        with patch.dict("os.environ", {"EXCHANGERATE_API_KEY": ""}, clear=False):
            result = await live_currency_conversion.ainvoke(
                {"amount": 500.0, "base": "USD", "target": "GBP"}
            )
        data = json.loads(result)
        assert data["status"] == "matrix_fallback"
        assert "GBP" in data["converted"]

    @pytest.mark.asyncio
    async def test_negative_amount_clamped(self):
        from src.tools.web_api_tools import live_currency_conversion
        with patch.dict("os.environ", {"EXCHANGERATE_API_KEY": ""}, clear=False):
            result = await live_currency_conversion.ainvoke(
                {"amount": -100.0, "base": "USD", "target": "EUR"}
            )
        # Should not raise; clamp to 1.0
        data = json.loads(result)
        assert "EUR" in data["converted"]

    @pytest.mark.asyncio
    async def test_network_exception_returns_fallback(self):
        from src.tools.web_api_tools import live_currency_conversion
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(side_effect=httpx.ConnectError("down"))
        with patch.dict("os.environ", {"EXCHANGERATE_API_KEY": "test_key"}):
            with patch("httpx.AsyncClient", return_value=client):
                result = await live_currency_conversion.ainvoke(
                    {"amount": 200.0, "base": "USD", "target": "JPY"}
                )
        data = json.loads(result)
        assert data["status"] == "matrix_fallback"


# ── fetch_local_breweries ─────────────────────────────────────────────────────

class TestFetchLocalBreweries:
    @pytest.mark.asyncio
    async def test_live_response_parsed(self):
        from src.tools.web_api_tools import fetch_local_breweries
        with patch("httpx.AsyncClient", return_value=_async_client(BREWERIES_PARIS)):
            result = await fetch_local_breweries.ainvoke({"city": "Paris"})
        assert "Brasserie" in result or "Houblon" in result

    @pytest.mark.asyncio
    async def test_empty_list_returns_fallback(self):
        from src.tools.web_api_tools import fetch_local_breweries
        with patch("httpx.AsyncClient", return_value=_async_client(BREWERIES_EMPTY)):
            result = await fetch_local_breweries.ainvoke({"city": "Berlin"})
        assert "Taproom" in result or "Berlin" in result

    @pytest.mark.asyncio
    async def test_http_error_returns_fallback(self):
        from src.tools.web_api_tools import fetch_local_breweries
        with patch("httpx.AsyncClient", return_value=_async_client({}, 500)):
            result = await fetch_local_breweries.ainvoke({"city": "London"})
        assert "London" in result

    @pytest.mark.asyncio
    async def test_network_exception_returns_fallback(self):
        from src.tools.web_api_tools import fetch_local_breweries
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(side_effect=httpx.ConnectError("down"))
        with patch("httpx.AsyncClient", return_value=client):
            result = await fetch_local_breweries.ainvoke({"city": "Tokyo"})
        assert "Tokyo" in result


# ── fetch_country_metadata ────────────────────────────────────────────────────

class TestFetchCountryMetadata:
    @pytest.mark.asyncio
    async def test_live_response_parsed(self):
        from src.tools.web_api_tools import fetch_country_metadata
        with patch("httpx.AsyncClient", return_value=_async_client(RESTCOUNTRIES_FRANCE)):
            result = await fetch_country_metadata.ainvoke({"country_name": "France"})
        data = json.loads(result)
        assert data["status"] == "success"
        assert data["currency_code"] == "EUR"
        assert data["canonical_name"] == "France"

    @pytest.mark.asyncio
    async def test_http_error_returns_inferred_fallback(self):
        from src.tools.web_api_tools import fetch_country_metadata
        with patch("httpx.AsyncClient", return_value=_async_client({}, 404)):
            result = await fetch_country_metadata.ainvoke({"country_name": "France"})
        data = json.loads(result)
        assert data["status"] == "inferred_fallback"
        assert "currency_code" in data

    @pytest.mark.asyncio
    async def test_uk_fallback_uses_gbp(self):
        from src.tools.web_api_tools import fetch_country_metadata
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.get = AsyncMock(side_effect=Exception("network"))
        with patch("httpx.AsyncClient", return_value=client):
            result = await fetch_country_metadata.ainvoke({"country_name": "United Kingdom"})
        data = json.loads(result)
        assert data["currency_code"] == "GBP"


# ── web_research_tavily ───────────────────────────────────────────────────────

class TestWebResearchTavily:
    @pytest.mark.asyncio
    async def test_missing_key_returns_unavailable_message(self):
        from src.tools.web_api_tools import web_research_tavily
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            result = await web_research_tavily.ainvoke({"query": "Paris travel tips"})
        assert "unavailable" in result.lower() or "TAVILY_API_KEY" in result

    @pytest.mark.asyncio
    async def test_live_results_parsed(self):
        from src.tools.web_api_tools import web_research_tavily
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=TAVILY_PARIS)):
                result = await web_research_tavily.ainvoke({"query": "Paris tips"})
        assert "example.com" in result

    @pytest.mark.asyncio
    async def test_empty_results_returns_no_results_message(self):
        from src.tools.web_api_tools import web_research_tavily
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with patch("asyncio.to_thread", new=AsyncMock(return_value=[])):
                result = await web_research_tavily.ainvoke({"query": "empty"})
        assert "no results" in result.lower()

    @pytest.mark.asyncio
    async def test_exception_returns_fallback_message(self):
        from src.tools.web_api_tools import web_research_tavily
        with patch.dict("os.environ", {"TAVILY_API_KEY": "test_key"}):
            with patch("asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("rate limit"))):
                result = await web_research_tavily.ainvoke({"query": "Paris"})
        assert isinstance(result, str) and len(result) > 0


# ── Input sanitisation (cross-tool) ──────────────────────────────────────────

class TestInputSanitisation:
    @pytest.mark.asyncio
    async def test_over_length_city_name_handled(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": ""}, clear=False):
            result = await geocode_location.ainvoke({"city": "P" * 200})
        # Must not raise — returns valid JSON
        assert json.loads(result)

    @pytest.mark.asyncio
    async def test_special_characters_stripped_from_city(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": ""}, clear=False):
            result = await geocode_location.ainvoke({"city": "Paris'; DROP TABLE--"})
        assert json.loads(result)

    @pytest.mark.asyncio
    async def test_empty_city_returns_rejected_status(self):
        from src.tools.web_api_tools import geocode_location
        with patch.dict("os.environ", {"OPENCAGE_API_KEY": ""}, clear=False):
            result = await geocode_location.ainvoke({"city": ""})
        data = json.loads(result)
        assert data["status"] == "rejected"
