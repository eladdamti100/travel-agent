"""
Plan formatter — deterministic section builders and text formatters for the final plan.

Extracted from planner.py. Contains no LLM calls — pure data-to-markdown logic.
"""

import datetime
import json
import re
from typing import Dict, List

from src.config.city_registry import CITY_BY_AIRPORT as _AIRPORT_CITY
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("plan_formatter")

# ── Noise filter ──────────────────────────────────────────────────────────────

_NOISE_PATTERNS = (
    "image ",
    "select dates", "weekly", "monthly", "view all", "load more",
    "read more", "click here", "sign up", "subscribe", "advertisement",
)

_CATEGORY_LABELS: Dict[str, str] = {
    "fetch_activities":      "Activities",
    "fetch_restaurants":     "Restaurants",
    "fetch_weather":         "Weather",
    "events_finder":         "Events",
    "local_transport_guide": "Local Transport",
    "airport_transfer_info": "Airport Transfers",
}


def is_noisy_line(text: str) -> bool:
    """Returns True when a line looks like UI junk or a scraping artefact."""
    low = text.strip().lower()
    if not low:
        return True
    if any(low.startswith(p) for p in _NOISE_PATTERNS):
        return True
    if low.count("|") >= 3:
        return True
    return False


# ── Section 1: DB data ────────────────────────────────────────────────────────

def build_db_section(context: TripContext, db_results: Dict[str, str], has_live_flights: bool = False) -> str:
    """Builds Section 1 (Database Data) deterministically from SQLite tool results."""
    parts: List[str] = ["# Section 1 — Database Data\n"]

    budget_str = (
        f"${context.total_budget:,.2f} {context.currency or 'USD'}"
        if context.total_budget else "—"
    )
    airport = context.origin_airport or "—"
    origin_city = _AIRPORT_CITY.get(airport.upper(), context.origin_country or "—")

    parts.append("**Trip Summary**")
    parts.append(f"- Origin: {airport} ({origin_city})")
    parts.append(f"- Destination: {context.destination_city or '—'}, {context.destination_country or '—'}")
    parts.append(f"- Duration: {context.duration_days or '—'} days")
    if context.travel_month:
        parts.append(f"- Travel month: {context.travel_month.capitalize()}")
    parts.append(f"- Total budget: {budget_str}")
    if context.travel_style:
        parts.append(f"- Travel style: {context.travel_style.capitalize()}")
    parts.append("")

    # Flights
    parts.append("**Flights**")
    flights_raw = db_results.get("fetch_flights", "")
    flights_is_web = flights_raw.startswith("[Web source]") if flights_raw else False
    live_flights_available = has_live_flights

    if live_flights_available:
        parts.append("- Live flight prices from Google Flights — see Section 2 below.")
    elif flights_raw and not flights_is_web:
        try:
            flights = json.loads(flights_raw)
            if isinstance(flights, list) and flights:
                for f in flights[:5]:
                    price = f.get("price")
                    price_str = f"${price}" if price else "price unavailable"
                    parts.append(
                        f"- {f.get('airline', '—')}: {f.get('flight_number', '—')}, {price_str}"
                    )
            else:
                parts.append("- No confirmed flight records in database — see web data below.")
        except (json.JSONDecodeError, TypeError):
            parts.append("- No confirmed flight records in database — see web data below.")
    else:
        parts.append("- No confirmed flight records in database — see web data below.")
    parts.append("")

    # Hotels
    parts.append("**Hotels**")
    hotels_raw = db_results.get("fetch_hotels", "")
    if hotels_raw:
        try:
            hotels = json.loads(hotels_raw)
            if isinstance(hotels, list) and hotels:
                for h in hotels[:5]:
                    price = h.get("price_per_night", "—")
                    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else f"${price}"
                    parts.append(
                        f"- {h.get('name', '—')}: {price_str}/night ({h.get('stars', '—')} stars)"
                    )
            else:
                parts.append(f"- {hotels_raw}")
        except (json.JSONDecodeError, TypeError):
            parts.append(f"- {hotels_raw}")
    else:
        parts.append("- Not available")
    parts.append("")

    # Activities & Experience
    parts.append("**Activities & Experience**")
    parts.append("")
    has_experience = False

    for raw_key, formatter in [
        ("fetch_activities",      fmt_activities),
        ("fetch_restaurants",     fmt_restaurants),
        ("fetch_weather",         fmt_weather),
        ("events_finder",         fmt_events),
        ("local_transport_guide", fmt_transport),
        ("airport_transfer_info", fmt_airport),
    ]:
        raw = db_results.get(raw_key, "")
        if raw:
            lines = formatter(raw)
            if lines:
                label = _CATEGORY_LABELS.get(raw_key, raw_key)
                parts.append(f"*{label}*")
                parts.extend(lines)
                parts.append("")
                has_experience = True

    if not has_experience:
        parts.append("- Not available")
        parts.append("")

    # Visa
    parts.append("**Visa Information**")
    visa_raw = db_results.get("check_visa", "")
    visa_is_web = visa_raw.startswith("[Web source]") if visa_raw else False
    if visa_raw and not visa_is_web:
        parts.append(f"- {visa_raw}")
    else:
        parts.append("- No visa data in database — see web research section below.")
    parts.append("")

    # Cost Summary
    parts.append("**Cost Summary**")
    cost_raw = db_results.get("calculate_trip_cost", "")
    flights_raw_for_cost = db_results.get("fetch_flights", "")
    live_raw_for_cost    = db_results.get("fetch_live_flights", "")
    flight_is_estimated  = not flights_raw_for_cost or flights_raw_for_cost.startswith("[Web source]")

    # If live flights exist but DB flights don't, add cheapest live price to cost summary
    if has_live_flights and not flights_raw_for_cost:
        try:
            live_list = json.loads(live_raw_for_cost) if live_raw_for_cost else []
            if live_list:
                cheapest = min(live_list, key=lambda f: f.get("price", 9999))
                parts.append(
                    f"- Cheapest flight (Google Flights): "
                    f"${cheapest.get('price')} — {cheapest.get('airline')} "
                    f"{cheapest.get('duration', '')}"
                )
                flight_is_estimated = False
        except (json.JSONDecodeError, TypeError):
            pass
    if cost_raw:
        try:
            cost = json.loads(cost_raw)
            for k, v in cost.items():
                if k == "currency":
                    continue
                label = k.replace("_", " ").title()
                if k == "flight" and flight_is_estimated:
                    parts.append(f"- {label}: {v} (flight price not confirmed — estimate only)")
                else:
                    parts.append(f"- {label}: {v}")
        except (json.JSONDecodeError, TypeError):
            parts.append(f"- {cost_raw}")
    else:
        parts.append("- Cost calculation not available")

    return "\n".join(parts)


# ── Row formatters ────────────────────────────────────────────────────────────

def fmt_activities(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        out = []
        for a in items[:5]:
            price = a.get("price")
            price_str = f"${price:.0f}" if isinstance(price, (int, float)) and price else "Free"
            out.append(f"- {a.get('name', '—')} ({a.get('category', '—')}): {price_str}")
        return out
    except (json.JSONDecodeError, TypeError):
        return [f"- {raw}"] if raw and not raw.startswith("No ") else []


def fmt_restaurants(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        return [
            f"- Restaurant: {r.get('name', '—')} ({r.get('cuisine', '—')}), "
            f"{r.get('price_range', '—')}, ★{r.get('rating', '—')}"
            for r in items[:3]
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def fmt_weather(raw: str) -> List[str]:
    try:
        w = json.loads(raw)
        month = (w.get("month") or "").capitalize()
        return [
            f"- Weather ({month}): {w.get('avg_temp_c', '—')}°C / "
            f"{w.get('avg_temp_f', '—')}°F — {w.get('description', '—')}"
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def fmt_events(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        return [
            f"- Event: {e.get('name', '—')} ({e.get('category', '—')})"
            for e in items[:3]
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def fmt_transport(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        return [
            f"- Local transport: {m.get('mode', '—')} — {m.get('price_range', '—')}"
            for m in items[:3]
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def fmt_airport(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        out = []
        for t in items[:2]:
            price = t.get("price_usd", "—")
            price_str = f"~${price:.0f}" if isinstance(price, (int, float)) else f"~${price}"
            out.append(
                f"- Airport transfer: {t.get('mode', '—')}, "
                f"{t.get('duration', '—')}, {price_str}"
            )
        return out
    except (json.JSONDecodeError, TypeError):
        return []


# ── Section 2: Web data ───────────────────────────────────────────────────────

def extract_visa_summary(raw: str) -> str:
    """Extracts one clean visa status sentence from raw web research text."""
    _VISA_KEYWORDS = ("visa-free", "no visa", "visa required", "etias", "visa on arrival")
    candidates = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("source:") or line.lower().startswith("content:"):
            continue
        if line.count("|") >= 2:
            continue
        low = line.lower()
        if any(kw in low for kw in _VISA_KEYWORDS):
            for s in re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                if (
                    len(s) > 20
                    and s.count("|") < 2
                    and any(kw in s.lower() for kw in _VISA_KEYWORDS)
                ):
                    candidates.append(s)
                    break
        if candidates:
            break

    if candidates:
        return candidates[0] + " (Source: web research)"
    return "Visa requirements could not be determined — check the official embassy website."


def build_web_section(web_results: Dict[str, str]) -> str:
    """Builds Section 2 (Live Web Data) deterministically from WebAgent results."""
    parts: List[str] = ["# Section 2 — Live Web Data\n"]
    has_any = False

    # Live flights from SerpAPI Google Flights
    live_flights_raw = web_results.get("fetch_live_flights", "")
    if live_flights_raw:
        try:
            live_flights = json.loads(live_flights_raw)
            if isinstance(live_flights, list) and live_flights:
                parts.append("**Live Flights (Google Flights)**")
                for f in live_flights[:5]:
                    price    = f.get("price", "?")
                    airline  = f.get("airline", "?")
                    duration = f.get("duration", "")
                    dur_str  = f"  |  {duration}" if duration else ""
                    parts.append(f"- {airline}: ${price}{dur_str}")
                parts.append("")
                has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    flights_web = web_results.get("fetch_flights", "")
    if flights_web and flights_web.startswith("[Web source]"):
        parts.append("**Flights (Web Research)**")
        parts.append("- No direct flight records found in database for this route.")
        parts.append("- Check current prices and availability on Google Flights, Skyscanner, or Kayak.")
        parts.append("")
        has_any = True

    visa_web = web_results.get("check_visa", "")
    if visa_web and visa_web.startswith("[Web source]"):
        clean = visa_web.removeprefix("[Web source]").strip()
        visa_line = extract_visa_summary(clean)
        parts.append("**Visa Information (Web Research)**")
        parts.append(f"- {visa_line}")
        parts.append("")
        has_any = True

    geo_raw = web_results.get("geocode_location", "")
    if geo_raw:
        try:
            geo = json.loads(geo_raw)
            source = " (live)" if geo.get("status") == "verified_live" else " (fallback)"
            parts.append(f"**Location Coordinates**{source}")
            parts.append(f"- Lat: {geo.get('lat', '—')}, Lng: {geo.get('lng', '—')}\n")
            has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    country_raw = web_results.get("fetch_country_metadata", "")
    if country_raw:
        try:
            meta = json.loads(country_raw)
            parts.append("**Country Metadata**")
            parts.append(f"- Country: {meta.get('canonical_name', '—')}")
            parts.append(f"- Local currency: {meta.get('currency_code', '—')}")
            parts.append(f"- Region: {meta.get('region', '—')}\n")
            has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    currency_raw = web_results.get("live_currency_conversion", "")
    if currency_raw:
        try:
            curr = json.loads(currency_raw)
            source = " (live)" if curr.get("status") == "live_synchronized" else " (estimate)"
            dest_cur = curr.get("destination_currency", "")
            pass_cur = curr.get("passport_currency", "")

            parts.append(f"**Currency Conversion**{source}")
            parts.append(f"- Budget: {curr.get('original', '—')} = {curr.get('converted', '—')}")

            inv = curr.get("inverse_rate")
            if inv and dest_cur:
                parts.append(f"- 1 {dest_cur} = {inv} USD")

            one_in_pass = curr.get("one_dest_in_passport")
            if one_in_pass and dest_cur and pass_cur and pass_cur != "USD":
                parts.append(f"- 1 {dest_cur} = {one_in_pass} ({pass_cur})")

            budget_in_pass = curr.get("budget_in_passport")
            if budget_in_pass and pass_cur and pass_cur != "USD":
                parts.append(f"- Full budget in {pass_cur}: {budget_in_pass}")

            parts.append("")
            has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    events_raw = web_results.get("fetch_live_events", "")
    if events_raw:
        event_lines = []
        for line in events_raw.strip().split("\n"):
            stripped = line.strip().lstrip("- ")
            if stripped and not is_noisy_line(stripped):
                event_lines.append(f"- {stripped}")
            if len(event_lines) >= 5:
                break
        if event_lines:
            parts.append("**Live Events**")
            parts.extend(event_lines)
            parts.append("")
            has_any = True

    brew_raw = web_results.get("fetch_breweries", "")
    if brew_raw:
        seen_names: set = set()
        brew_lines = []
        for line in brew_raw.strip().split("\n"):
            stripped = line.strip().lstrip("- ")
            if not stripped or is_noisy_line(stripped):
                continue
            name_key = stripped.split("|")[0].strip().lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            brew_lines.append(f"- {stripped}")
            if len(brew_lines) >= 3:
                break
        if brew_lines:
            parts.append("**Local Breweries & Pubs**")
            parts.extend(brew_lines)
            parts.append("")
            has_any = True

    # Local transport (structured JSON from fetch_local_transport_live)
    transport_raw = web_results.get("fetch_local_transport", "")
    if not transport_raw:
        transport_raw = web_results.get("local_transport_guide", "")
    if transport_raw:
        try:
            t = json.loads(transport_raw)
            if isinstance(t, dict) and "options" in t:
                source_tag = " (live enriched)" if "live" in t.get("source", "") else ""
                parts.append(f"**Local Transport — {t.get('city', 'City')}{source_tag}**")

                at = t.get("airport_transfer", {})
                if at:
                    price = at.get("price_usd")
                    price_str = f"  ~${price}" if price else ""
                    dur = at.get("duration_min")
                    dur_str = f"  {dur} min" if dur else ""
                    parts.append(
                        f"- Airport transfer: **{at.get('mode', '?')}** "
                        f"({at.get('from', '')} → {at.get('to', '')})"
                        f"{dur_str}{price_str}"
                    )
                    if at.get("tip"):
                        parts.append(f"  → *{at['tip']}*")

                metro = t.get("metro", {})
                if metro:
                    parts.append(
                        f"- Metro/Subway: **{metro.get('network', '?')}** "
                        f"— ${metro.get('single_usd', '?')}/ride  |  "
                        f"Day pass ${metro.get('day_pass_usd', '?')}"
                    )
                    if metro.get("card"):
                        parts.append(f"  → Card: {metro['card']}")

                for opt in t.get("options", [])[:4]:
                    mode     = opt.get("mode", "?")
                    price    = opt.get("price", "")
                    day_pass = opt.get("day_pass", "")
                    notes    = opt.get("notes", "")
                    dp_str   = f"  day pass {day_pass}" if day_pass else ""
                    parts.append(f"- {mode}: {price}{dp_str}  — {notes}")

                if t.get("live_update"):
                    parts.append(f"\n  *Live update:* {t['live_update'][:200]}")

                parts.append("")
                has_any = True
        except (json.JSONDecodeError, TypeError):
            if transport_raw and len(transport_raw) > 10:
                parts.append("**Local Transport**")
                parts.append(f"- {transport_raw[:400]}")
                parts.append("")
                has_any = True

    tavily_raw = web_results.get("web_research_tavily", "")
    if tavily_raw and not tavily_raw.startswith("Search Engine"):
        bullets = parse_tavily_bullets(tavily_raw)
        if bullets:
            parts.append("**Web Research Highlights**")
            parts.extend(bullets)
            parts.append("")
            has_any = True

    if not has_any:
        parts.append("- No live web data was available for this trip.")

    return "\n".join(parts)


def parse_tavily_bullets(raw: str, max_bullets: int = 4) -> List[str]:
    """Strips noise and returns clean bullet-point strings from a Tavily result."""
    _today = datetime.date.today()
    _past_year_re = re.compile(r"\b(20\d{2})\b")

    def _sentence_is_stale(s: str) -> bool:
        years = [int(y) for y in _past_year_re.findall(s)]
        if not years:
            return False
        return all(y < _today.year - 1 for y in years)

    sentences: List[str] = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            line = line.strip()
            if not line or line.lower().startswith("source:"):
                continue
            if line.lower().startswith("content:"):
                line = line[len("content:"):].strip()
            if is_noisy_line(line):
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                if len(s) > 30 and not is_noisy_line(s) and not _sentence_is_stale(s):
                    sentences.append(s)

    seen: set = set()
    bullets: List[str] = []
    for s in sentences:
        key = s[:60].lower()
        if key not in seen:
            seen.add(key)
            bullets.append(f"- {s}")
        if len(bullets) >= max_bullets:
            break

    return bullets
