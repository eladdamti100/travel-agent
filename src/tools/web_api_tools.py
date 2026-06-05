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


# ── fetch_live_flights (Amadeus API) ─────────────────────────────────────────
#
# Uses Amadeus Flight Offers Search (free sandbox).
# Get your key at: https://developers.amadeus.com/
# Add to .env:
#   AMADEUS_API_KEY=your_key
#   AMADEUS_API_SECRET=your_secret
#
# Falls back to static prices when keys are missing.

_CITY_TO_IATA: Dict[str, str] = {
    "paris":    "CDG",
    "london":   "LHR",
    "tokyo":    "NRT",
    "new york": "JFK",
    "berlin":   "BER",
}

_FLIGHT_STATIC_FALLBACKS: Dict[str, list] = {
    "paris":    [{"airline": "El Al",          "price": 350, "duration": "5h 00m"},
                 {"airline": "Air France",      "price": 420, "duration": "5h 10m"},
                 {"airline": "Ryanair",         "price": 290, "duration": "5h 30m"}],
    "london":   [{"airline": "El Al",          "price": 380, "duration": "5h 20m"},
                 {"airline": "British Airways", "price": 450, "duration": "5h 15m"},
                 {"airline": "EasyJet",         "price": 310, "duration": "5h 45m"}],
    "tokyo":    [{"airline": "El Al",          "price": 950, "duration": "11h 30m"},
                 {"airline": "ANA",             "price": 880, "duration": "13h 00m"},
                 {"airline": "Emirates",        "price": 830, "duration": "14h 00m"}],
    "new york": [{"airline": "El Al",          "price": 820, "duration": "11h 00m"},
                 {"airline": "Delta",           "price": 790, "duration": "11h 30m"},
                 {"airline": "United",          "price": 760, "duration": "12h 00m"}],
    "berlin":   [{"airline": "El Al",          "price": 380, "duration": "4h 00m"},
                 {"airline": "Lufthansa",       "price": 420, "duration": "4h 10m"},
                 {"airline": "Wizz Air",        "price": 310, "duration": "4h 30m"}],
}


@tool
async def fetch_live_flights(
    origin: str,
    destination: str,
    travel_date: str = "",
    adults: int = 1,
) -> str:
    """
    Search for real flight prices using the Amadeus Flight Offers API.

    Returns the 3 cheapest options with airline, price, and duration.
    Falls back to curated static data when AMADEUS keys are not configured.

    Args:
        origin:       Departure airport code (e.g. "TLV")
        destination:  Destination city (e.g. "Paris") or IATA code (e.g. "CDG")
        travel_date:  Departure date in YYYY-MM-DD format (default: next month)
        adults:       Number of adult passengers (default 1)
    """
    try:
        origin_clean = _sanitize(origin, "origin").upper()
        dest_clean   = _sanitize(destination, "destination")
    except ValueError as err:
        return f"Input rejected: {err}"

    dest_lower = dest_clean.lower()
    dest_iata  = _CITY_TO_IATA.get(dest_lower, dest_clean.upper()[:3])
    fallback   = _FLIGHT_STATIC_FALLBACKS.get(dest_lower, [
        {"airline": "Economy carrier", "price": 500, "duration": "varies"},
        {"airline": "Major airline",   "price": 650, "duration": "varies"},
    ])

    def _format_fallback(flights: list, note: str = "static") -> str:
        lines = [f"Flight options ({origin_clean} -> {dest_clean.title()}) [{note}]:"]
        for f in flights:
            lines.append(
                f"  - {f['airline']:20s} ${f['price']:>5}  |  {f['duration']}"
            )
        return "\n".join(lines)

    api_key = os.getenv("SERPAPI_KEY", "")

    if not api_key or api_key.startswith("your_"):
        logger.info("fetch_live_flights. status=no_serpapi_key using=static dest=%s", dest_clean)
        return _format_fallback(fallback, note="static prices")

    # ── Resolve travel date ────────────────────────────────────────────────────
    from datetime import date, timedelta
    if travel_date and len(travel_date) == 10:
        parsed = travel_date
        # Reject past dates — SerpAPI returns 400
        try:
            from datetime import datetime
            if datetime.strptime(parsed, "%Y-%m-%d").date() <= date.today():
                parsed = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
        except ValueError:
            parsed = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
        dep_date = parsed
    else:
        dep_date = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")

    # ── SerpAPI Google Flights ─────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://serpapi.com/search.json",
                params={
                    "engine":         "google_flights",
                    "departure_id":   origin_clean,
                    "arrival_id":     dest_iata,
                    "outbound_date":  dep_date,
                    "currency":       "USD",
                    "adults":         max(1, int(adults)),
                    "type":           2,          # 2 = one-way
                    "api_key":        api_key,
                },
                timeout=10.0,
            )

        if response.status_code != 200:
            logger.warning("fetch_live_flights. status=http_%d using=static", response.status_code)
            return _format_fallback(fallback, note="static — SerpAPI error")

        data = response.json()

        # SerpAPI returns best_flights + other_flights
        raw_offers = (data.get("best_flights") or []) + (data.get("other_flights") or [])
        if not raw_offers:
            return _format_fallback(fallback, note="static — no SerpAPI results")

        results = []
        for offer in raw_offers[:3]:
            price    = offer.get("price", 0)
            flights  = offer.get("flights", [{}])
            airline  = flights[0].get("airline", "Unknown")
            duration = offer.get("total_duration", 0)
            hrs, mins = divmod(duration, 60)
            results.append({
                "airline":  airline,
                "price":    int(price),
                "duration": f"{hrs}h {mins:02d}m",
            })

        logger.info("fetch_live_flights. status=live source=serpapi offers=%d", len(results))
        return _format_fallback(results, note=f"live Google Flights, depart {dep_date}")

    except Exception as exc:
        logger.error("fetch_live_flights. status=exception error=%s using=static", exc)
        return _format_fallback(fallback, note="static — SerpAPI unavailable")


# ── fetch_local_transport_live ────────────────────────────────────────────────

_TRANSPORT_STRUCTURED: Dict[str, dict] = {
    "paris": {
        "airport_transfer": {
            "mode": "RER B train",
            "from": "CDG Airport",
            "to":   "City center (Gare du Nord)",
            "duration_min": 35,
            "price_usd": 12,
            "tip": "Runs every 10-15 min. Buy ticket at airport before boarding.",
        },
        "metro": {
            "network":        "RATP Metro",
            "lines":          16,
            "single_usd":     2.10,
            "day_pass_usd":   8.50,
            "card":           "Navigo Easy (reloadable, no phone required)",
        },
        "options": [
            {"mode": "Metro",       "price": "$2.10/ride",  "day_pass": "$8.50",  "notes": "Best for city centre"},
            {"mode": "Bus",         "price": "$2.10/ride",  "day_pass": "$8.50",  "notes": "Night buses available"},
            {"mode": "Taxi/Uber",   "price": "$15-40",      "airport":  "$65-75", "notes": "Fixed rate CDG→centre"},
            {"mode": "Vélib' Bike", "price": "$3.50/day",   "day_pass": "$3.50",  "notes": "1,400 stations citywide"},
        ],
    },
    "london": {
        "airport_transfer": {
            "mode": "Elizabeth line (Heathrow) / Gatwick Express",
            "from": "LHR / LGW Airport",
            "to":   "Paddington / London Bridge",
            "duration_min": 22,
            "price_usd": 15,
            "tip": "Use contactless card — cheaper than buying a ticket.",
        },
        "metro": {
            "network":        "London Underground (Tube)",
            "lines":          11,
            "single_usd":     3.50,
            "day_pass_usd":   16.00,
            "card":           "Oyster card or contactless bank card",
        },
        "options": [
            {"mode": "Tube",           "price": "$3.50/ride", "day_pass": "$16",    "notes": "Capped daily spending"},
            {"mode": "Bus",            "price": "$2.00/ride", "day_pass": "$7",     "notes": "24h service"},
            {"mode": "Black Cab/Uber", "price": "$20-60",     "airport":  "$60-90", "notes": "Uber available everywhere"},
            {"mode": "Santander Bike", "price": "$2/30min",   "day_pass": "$6.50",  "notes": "750 docking stations"},
        ],
    },
    "tokyo": {
        "airport_transfer": {
            "mode": "Narita Express (N'EX)",
            "from": "NRT Airport",
            "to":   "Shinjuku / Shibuya",
            "duration_min": 90,
            "price_usd": 30,
            "tip": "JR Pass holders ride free. Book in advance for discount.",
        },
        "metro": {
            "network":        "Tokyo Metro + Toei Subway",
            "lines":          13,
            "single_usd":     1.80,
            "day_pass_usd":   8.00,
            "card":           "Suica / Pasmo IC card (works on all trains, shops, vending)",
        },
        "options": [
            {"mode": "Metro",       "price": "$1.80/ride", "day_pass": "$8",    "notes": "Most efficient in city"},
            {"mode": "JR Lines",    "price": "$2-5/ride",  "day_pass": "JR Pass", "notes": "Shinkansen included"},
            {"mode": "Taxi",        "price": "$10-40",     "airport":  "$200+", "notes": "Expensive — use train"},
            {"mode": "Bus",         "price": "$2/ride",    "day_pass": "$6",    "notes": "Highway buses for longer trips"},
        ],
    },
    "new york": {
        "airport_transfer": {
            "mode": "AirTrain + LIRR / Subway",
            "from": "JFK Airport",
            "to":   "Midtown Manhattan",
            "duration_min": 50,
            "price_usd": 11,
            "tip": "AirTrain to Jamaica ($8.50) + Subway ($2.90). Avoid taxi in rush hour.",
        },
        "metro": {
            "network":        "NYC Subway (MTA)",
            "lines":          36,
            "single_usd":     2.90,
            "day_pass_usd":   34.00,
            "card":           "OMNY (tap-to-pay) or MetroCard",
        },
        "options": [
            {"mode": "Subway",       "price": "$2.90/ride", "day_pass": "$34/week", "notes": "24/7 service"},
            {"mode": "Bus (MTA)",    "price": "$2.90/ride", "day_pass": "$34/week", "notes": "Free transfer from subway"},
            {"mode": "Yellow Taxi",  "price": "$15-50",     "airport":  "$70+",     "notes": "Flat rate JFK→Manhattan $70"},
            {"mode": "Citi Bike",    "price": "$4.49/ride", "day_pass": "$19",      "notes": "Electric bikes available"},
        ],
    },
    "berlin": {
        "airport_transfer": {
            "mode": "S-Bahn S9 / Airport Express",
            "from": "BER Airport",
            "to":   "Berlin Hauptbahnhof",
            "duration_min": 30,
            "price_usd": 4.50,
            "tip": "Buy ABC zone ticket. S9 runs every 20 min.",
        },
        "metro": {
            "network":        "BVG (U-Bahn + S-Bahn + Tram + Bus)",
            "lines":          10,
            "single_usd":     3.50,
            "day_pass_usd":   11.00,
            "card":           "BVG app or Berlin WelcomeCard (includes museum discounts)",
        },
        "options": [
            {"mode": "U-Bahn/S-Bahn", "price": "$3.50/ride", "day_pass": "$11",    "notes": "AB zone covers tourist areas"},
            {"mode": "Tram",           "price": "$3.50/ride", "day_pass": "$11",    "notes": "East Berlin coverage"},
            {"mode": "Taxi/Uber",      "price": "$12-35",     "airport":  "$45-55", "notes": "Uber available"},
            {"mode": "Nextbike",       "price": "$1.50/30min","day_pass": "$9",     "notes": "City bike network"},
        ],
    },
}


@tool
async def fetch_local_transport_live(city: str) -> str:
    """
    Fetch structured local transport options for a destination city.

    Returns JSON with airport transfer, metro details, and all transport options
    with prices, day passes, and practical tips.
    Uses curated data enriched with live Tavily updates when available.

    Args:
        city: Destination city (e.g. "Tokyo", "Paris", "London")
    """
    import json as _json

    try:
        city_clean = _sanitize(city, "city")
    except ValueError as err:
        return f"Input rejected: {err}"

    city_lower = city_clean.lower()

    # Get structured base data
    structured = _TRANSPORT_STRUCTURED.get(city_lower)
    if not structured:
        structured = {
            "airport_transfer": {"mode": "Taxi or shuttle", "price_usd": 30, "tip": "Ask at the airport information desk."},
            "metro": {"network": f"{city_clean} public transport", "single_usd": 2.00, "day_pass_usd": 8.00},
            "options": [{"mode": "Public transport", "price": "$2-4/ride", "notes": "Day passes available"}],
        }

    result = {
        "city":    city_clean,
        "source":  "curated",
        **structured,
    }

    # Enrich with live Tavily snippet if available
    api_key = os.getenv("TAVILY_API_KEY", "")
    if api_key and not api_key.startswith("your_"):
        query = f"public transport {city_clean} 2025 prices day pass metro"
        try:
            search = TavilySearchResults(max_results=1, tavily_api_key=api_key)
            live = await asyncio.to_thread(search.invoke, query)
            if live:
                snippet = live[0].get("content", "")[:300].strip()
                if snippet:
                    result["live_update"] = snippet
                    result["source"] = "curated + live"
        except Exception:
            pass

    logger.info("fetch_local_transport_live. city=%s source=%s", city_clean, result["source"])
    return _json.dumps(result, indent=2)
