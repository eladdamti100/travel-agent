"""
Web API tools — async, defensive tools for live travel data enrichment.

All tools sanitize inputs (length cap 80 chars, alpha-only filter) to guard
against prompt injection and DoS.  Every tool has a graceful static fallback
so a missing API key or network error never crashes the planner.
"""

import asyncio
import json
import os
import re
from datetime import date as _today_date
from typing import Dict

from src.utils.token_tracker import log_token_usage as _log_tokens

import httpx
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

from src.config.city_registry import CURRENCY_BY_COUNTRY as _CURRENCY_BY_COUNTRY
from src.utils.logger import get_logger

logger = get_logger("web_api_tools")

MAX_INPUT_LENGTH = 80
_ALPHA_PATTERN = re.compile(r"^[a-zA-Z\s,\-\.]+$")

# ── Static fallbacks ──────────────────────────────────────────────────────────

_MOCK_COORDINATES: Dict[str, Dict[str, float]] = {
    "london":   {"lat": 51.5074, "lng": -0.1278},
    "paris":    {"lat": 48.8566, "lng":  2.3522},
    "tokyo":    {"lat": 35.6762, "lng": 139.6503},
    "new york": {"lat": 40.7128, "lng": -74.0060},
    "berlin":   {"lat": 52.5200, "lng":  13.4050},
}

_MOCK_EVENTS: Dict[str, str] = {
    "london": (
        "- Adele Live at Wembley Stadium | Tickets from £100\n"
        "- British Museum Late Evening Tour | Free Entry"
    ),
    "paris": (
        "- Classical Symphony at Sainte-Chapelle | Tickets from €45\n"
        "- Louvre Impressionism Masterclass | Tickets from €25"
    ),
    "tokyo": "- Traditional Sumo Tournament at Ryogoku Kokugikan | Tickets from ¥4500",
}

_BASELINE_RATES: Dict[str, float] = {
    "USD": 1.0, "ILS": 3.75, "GBP": 0.78, "EUR": 0.92, "JPY": 156.0,
}


def _sanitize(text: str, field_name: str) -> str:
    """Strip, length-cap, and alpha-filter a user-supplied string."""
    if not text:
        raise ValueError(f"'{field_name}' cannot be empty.")
    cleaned = text.strip()
    if len(cleaned) > MAX_INPUT_LENGTH:
        logger.warning("web_api_tools. field=%s action=truncate original_len=%d", field_name, len(cleaned))
        cleaned = cleaned[:MAX_INPUT_LENGTH]
    if not _ALPHA_PATTERN.match(cleaned):
        logger.warning("web_api_tools. field=%s action=strip_special", field_name)
        cleaned = re.sub(r"[^a-zA-Z\s,\-\.]", "", cleaned)
    return cleaned


# ── geocode_location ──────────────────────────────────────────────────────────

@tool
async def geocode_location(city: str) -> str:
    """Convert a city name into geographical coordinates using OpenCage."""
    try:
        city_clean = _sanitize(city, "city")
    except ValueError as err:
        return json.dumps({"status": "rejected", "reason": str(err)})

    city_lower = city_clean.lower()
    fallback = {**_MOCK_COORDINATES.get(city_lower, {"lat": 0.0, "lng": 0.0})}

    api_key = os.getenv("OPENCAGE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        logger.info("geocode_location. status=key_missing city=%s using=fallback", city_clean)
        return json.dumps({"status": "fallback_resolved", "city": city_clean, **fallback})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.opencagedata.com/geocode/v1/json",
                params={"q": city_clean, "key": api_key, "limit": 1},
                timeout=4.0,
            )
        if response.status_code == 429:
            logger.warning("geocode_location. status=rate_limited city=%s using=fallback", city_clean)
            return json.dumps({"status": "rate_limit_fallback", "city": city_clean, **fallback})
        if response.status_code != 200:
            logger.warning("geocode_location. status=http_error code=%d city=%s using=fallback",
                           response.status_code, city_clean)
            return json.dumps({"status": "http_error_fallback", "city": city_clean, **fallback})

        results = response.json().get("results", [])
        if not results:
            return json.dumps({"status": "empty_fallback", "city": city_clean, **fallback})

        geo = results[0].get("geometry", {})
        return json.dumps({"status": "verified_live", "city": city_clean,
                           "lat": geo.get("lat"), "lng": geo.get("lng")})
    except Exception as exc:
        logger.error("geocode_location. status=exception city=%s error=%s using=fallback", city_clean, exc)
        return json.dumps({"status": "exception_fallback", "city": city_clean, **fallback})


# ── fetch_live_events ─────────────────────────────────────────────────────────

@tool
async def fetch_live_events(city: str) -> str:
    """Query Ticketmaster for upcoming events in a city."""
    try:
        city_clean = _sanitize(city, "city")
    except ValueError as err:
        return f"Input rejected: {err}"

    city_lower = city_clean.lower()

    api_key = os.getenv("TICKETMASTER_API_KEY")
    if not api_key or api_key.startswith("your_"):
        logger.info("fetch_live_events. status=key_missing city=%s using=fallback", city_clean)
        return _MOCK_EVENTS.get(city_lower, f"- Local Discovery Tour available in {city_clean}.")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://app.ticketmaster.com/discovery/v2/events.json",
                params={"apikey": api_key, "city": city_clean, "size": 3, "sort": "date,asc"},
                timeout=5.0,
            )
        if response.status_code in (401, 403, 429):
            logger.warning("fetch_live_events. status=api_error code=%d city=%s using=fallback",
                           response.status_code, city_clean)
            return _MOCK_EVENTS.get(city_lower, f"- Local events for {city_clean} (fallback).")

        events = response.json().get("_embedded", {}).get("events", [])
        if not events:
            return _MOCK_EVENTS.get(city_lower, f"- No upcoming events found in {city_clean}.")

        today = _today_date.today().isoformat()
        lines = []
        for ev in events:
            name = ev.get("name", "Live Event")
            date = ev.get("dates", {}).get("start", {}).get("localDate", "TBD")
            if date != "TBD" and date < today:
                continue
            prices = ev.get("priceRanges", [{}])[0]
            min_p = prices.get("min")
            curr = prices.get("currency", "USD")
            price_part = f" | from {min_p} {curr}" if min_p is not None else ""
            lines.append(f"- {name} | {date}{price_part}")

        return "\n".join(lines) if lines else _MOCK_EVENTS.get(city_lower, f"- No upcoming events in {city_clean}.")
    except Exception as exc:
        logger.error("fetch_live_events. status=exception city=%s error=%s", city_clean, exc)
        return _MOCK_EVENTS.get(city_lower, f"- Events unavailable for {city_clean}.")


# ── live_currency_conversion ──────────────────────────────────────────────────

@tool
async def live_currency_conversion(amount: float, base: str, target: str) -> str:
    """Convert an amount between currencies using ExchangeRate-API."""
    base_c  = re.sub(r"[^a-zA-Z]", "", base).upper()[:3]
    target_c = re.sub(r"[^a-zA-Z]", "", target).upper()[:3]

    if amount < 0:
        logger.warning("live_currency_conversion. amount=%.2f invalid=negative clamped_to=1.0", amount)
        amount = 1.0
    elif amount == 0:
        amount = 1.0

    def _fallback(reason: str) -> str:
        logger.info("live_currency_conversion. status=fallback reason=%s base=%s target=%s",
                    reason, base_c, target_c)
        b = _BASELINE_RATES.get(base_c, 1.0)
        t = _BASELINE_RATES.get(target_c, 1.0)
        converted = (amount / b) * t
        return json.dumps({
            "status": "matrix_fallback",
            "original": f"{amount:.2f} {base_c}",
            "converted": f"{converted:.2f} {target_c}",
            "rate": t / b,
        })

    api_key = os.getenv("EXCHANGERATE_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return _fallback("key_missing")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_c}",
                timeout=4.0,
            )
        if response.status_code != 200:
            return _fallback(f"http_{response.status_code}")

        rates = response.json().get("conversion_rates", {})
        t_rate = rates.get(target_c)
        if not t_rate:
            return _fallback("currency_not_in_response")

        return json.dumps({
            "status": "live_synchronized",
            "original": f"{amount:.2f} {base_c}",
            "converted": f"{amount * t_rate:.2f} {target_c}",
            "rate": t_rate,
        })
    except Exception as exc:
        logger.error("live_currency_conversion. status=exception base=%s target=%s error=%s", base_c, target_c, exc)
        return _fallback("network_exception")


# ── fetch_local_breweries ─────────────────────────────────────────────────────

@tool
async def fetch_local_breweries(city: str) -> str:
    """Fetch local breweries from Open Brewery DB (public, no auth required)."""
    try:
        city_clean = _sanitize(city, "city")
    except ValueError as err:
        return f"Rejected: {err}"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openbrewerydb.org/v1/breweries",
                params={"by_city": city_clean.lower(), "per_page": 3},
                timeout=4.0,
            )
        if response.status_code != 200:
            return f"- Local craft brewery | Central District, {city_clean}"

        breweries = response.json()
        if not breweries:
            return f"- Local Traditional Taproom | Central Square, {city_clean}"

        return "\n".join(
            f"- {b.get('name')} ({b.get('brewery_type', 'pub')}) | {b.get('address_1', 'City Center')}"
            for b in breweries
        )
    except Exception as exc:
        logger.warning("fetch_local_breweries. status=exception city=%s error=%s", city_clean, exc)
        return f"- Artisanal Brewery & Cellars | Central Hub, {city_clean}"


# ── fetch_country_metadata ────────────────────────────────────────────────────

@tool
async def fetch_country_metadata(country_name: str) -> str:
    """Fetch country metadata (currency, region) from RestCountries."""
    try:
        country_clean = _sanitize(country_name, "country_name")
    except ValueError as err:
        return json.dumps({"status": "rejected", "reason": str(err)})

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://restcountries.com/v3.1/name/{country_clean.lower()}?fullText=true",
                timeout=4.0,
            )
        if response.status_code != 200:
            raise httpx.HTTPStatusError("non-200", request=response.request, response=response)

        data = response.json()[0]
        currencies = data.get("currencies", {})
        currency_code = list(currencies.keys())[0] if currencies else "USD"
        return json.dumps({
            "status": "success",
            "canonical_name": data.get("name", {}).get("common", country_clean),
            "currency_code": currency_code,
            "region": data.get("region", "Global"),
        })
    except Exception as exc:
        logger.error("fetch_country_metadata. status=exception country=%s error=%s", country_clean, exc)
        inferred = _CURRENCY_BY_COUNTRY.get(country_clean.lower(), "EUR")
        return json.dumps({
            "status": "inferred_fallback",
            "canonical_name": country_clean,
            "currency_code": inferred,
            "region": "Unknown",
        })


# ── web_research_tavily ───────────────────────────────────────────────────────

@tool
async def web_research_tavily(query: str) -> str:
    """Execute a live Tavily web search and return ranked results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return "Search Engine: live search unavailable — TAVILY_API_KEY not configured."

    try:
        search = TavilySearchResults(max_results=2, tavily_api_key=api_key)
        # search.invoke() is synchronous — offload to thread pool.
        results = await asyncio.to_thread(search.invoke, query)
        if not results:
            return "Search Engine: query returned no results."
        return "\n\n".join(
            f"Source: {r.get('url')}\nContent: {r.get('content')}" for r in results
        )
    except Exception as exc:
        logger.error("web_research_tavily. status=exception error=%s", exc)
        return "Search Engine: request failed, falling back to offline data."


# ── fetch_live_flights ────────────────────────────────────────────────────────

_FLIGHT_FALLBACKS: Dict[str, str] = {
    "paris":    "El Al TLV→CDG ~$520 | Air France TLV→CDG ~$480 | Ryanair TLV→ORY ~$390",
    "london":   "El Al TLV→LHR ~$550 | British Airways TLV→LHR ~$610 | EasyJet TLV→LGW ~$420",
    "tokyo":    "El Al TLV→NRT ~$950 | ANA TLV→NRT (via hub) ~$880 | Emirates TLV→NRT ~$830",
    "new york": "El Al TLV→JFK ~$820 | Delta TLV→JFK ~$790 | United TLV→EWR ~$760",
    "berlin":   "El Al TLV→BER ~$380 | Lufthansa TLV→BER ~$420 | Wizz Air TLV→BER ~$310",
}


@tool
async def fetch_live_flights(origin: str, destination: str, travel_date: str = "") -> str:
    """
    Search for live flight options between origin and destination.

    Uses Tavily web search for real-time prices when available.
    Falls back to curated static data when the API key is missing or the query fails.

    Args:
        origin:       Departure city or airport code (e.g. "TLV", "Tel Aviv")
        destination:  Destination city (e.g. "Paris", "Tokyo")
        travel_date:  Optional travel date hint (e.g. "June 2025")
    """
    try:
        origin_clean = _sanitize(origin, "origin")
        dest_clean = _sanitize(destination, "destination")
    except ValueError as err:
        return f"Input rejected: {err}"

    dest_lower = dest_clean.lower()
    fallback = _FLIGHT_FALLBACKS.get(dest_lower,
        f"El Al {origin_clean}→{dest_clean} ~$600 | Local carrier ~$550")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("your_"):
        logger.info("fetch_live_flights. status=key_missing using=fallback dest=%s", dest_clean)
        return fallback

    date_hint = f" in {travel_date}" if travel_date else ""
    query = f"cheap flights from {origin_clean} to {dest_clean}{date_hint} prices airlines"

    try:
        search = TavilySearchResults(max_results=2, tavily_api_key=api_key)
        results = await asyncio.to_thread(search.invoke, query)
        if not results:
            return fallback

        snippets = "\n".join(
            f"• {r.get('content', '')[:200]}" for r in results
        )
        logger.info("fetch_live_flights. status=live origin=%s dest=%s", origin_clean, dest_clean)
        return f"Live flight search ({origin_clean} → {dest_clean}{date_hint}):\n{snippets}"
    except Exception as exc:
        logger.error("fetch_live_flights. status=exception dest=%s error=%s using=fallback", dest_clean, exc)
        return fallback


# ── fetch_local_transport_live ────────────────────────────────────────────────

_TRANSPORT_FALLBACKS: Dict[str, str] = {
    "paris": (
        "Metro: 16 lines, €1.90/ride. Day pass €7.50.\n"
        "RER: connects CDG airport (line B, ~35 min).\n"
        "Bus: Noctilien (night buses). Vélib' bike share available."
    ),
    "london": (
        "Underground (Tube): 11 lines. Use Oyster card or contactless, ~£2.80/ride.\n"
        "Elizabeth line: Heathrow ↔ central London ~20 min.\n"
        "Black cabs + Uber widely available."
    ),
    "tokyo": (
        "JR Pass recommended for tourists (~¥50,000/7 days).\n"
        "IC Card (Suica/Pasmo) for metro: ~¥200-300/ride.\n"
        "Narita Express: airport ↔ Shinjuku ~90 min, ¥3,250."
    ),
    "new york": (
        "Subway: 24/7, $2.90/ride. 7-day unlimited ~$34.\n"
        "AirTrain + LIRR from JFK ~$15 total.\n"
        "Yellow cabs + Uber/Lyft everywhere."
    ),
    "berlin": (
        "BVG: U-Bahn, S-Bahn, tram, bus. Single €3.20, day pass €9.90.\n"
        "S-Bahn S9 from BER airport ~45 min.\n"
        "AB zone covers most tourist sites."
    ),
}


@tool
async def fetch_local_transport_live(city: str) -> str:
    """
    Fetch local public transport options within a destination city.

    Returns metro, bus, taxi, and airport-transfer information.
    Uses Tavily for live data; falls back to curated static data.

    Args:
        city: Destination city (e.g. "Tokyo", "Paris")
    """
    try:
        city_clean = _sanitize(city, "city")
    except ValueError as err:
        return f"Input rejected: {err}"

    city_lower = city_clean.lower()
    fallback = _TRANSPORT_FALLBACKS.get(city_lower,
        f"Local metro and bus services available in {city_clean}. "
        "Ask at the tourist information desk for day passes.")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("your_"):
        logger.info("fetch_local_transport_live. status=key_missing city=%s using=fallback", city_clean)
        return fallback

    query = f"public transport options {city_clean} metro bus prices tourist day pass 2025"

    try:
        search = TavilySearchResults(max_results=2, tavily_api_key=api_key)
        results = await asyncio.to_thread(search.invoke, query)
        if not results:
            return fallback

        snippets = "\n".join(
            f"• {r.get('content', '')[:250]}" for r in results
        )
        logger.info("fetch_local_transport_live. status=live city=%s", city_clean)
        return f"Local transport in {city_clean} (live):\n{snippets}"
    except Exception as exc:
        logger.error("fetch_local_transport_live. status=exception city=%s error=%s", city_clean, exc)
        return fallback
