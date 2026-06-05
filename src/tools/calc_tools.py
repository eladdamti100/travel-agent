"""
Calculation tools — pure arithmetic, no database or API calls.
"""

import json
from typing import Optional

from langchain_core.tools import tool

# Flight distances in km between supported city pairs (symmetric)
_FLIGHT_DISTANCES_KM: dict = {
    frozenset({"tel aviv", "paris"}):    3310,
    frozenset({"tel aviv", "london"}):   3580,
    frozenset({"tel aviv", "tokyo"}):    9560,
    frozenset({"tel aviv", "new york"}): 9130,
    frozenset({"tel aviv", "berlin"}):   2830,
    frozenset({"paris",    "london"}):    340,
    frozenset({"paris",    "tokyo"}):    9720,
    frozenset({"paris",    "new york"}): 5840,
    frozenset({"paris",    "berlin"}):   1050,
    frozenset({"london",   "tokyo"}):    9540,
    frozenset({"london",   "new york"}): 5540,
    frozenset({"london",   "berlin"}):    930,
    frozenset({"tokyo",    "new york"}): 10800,
    frozenset({"tokyo",    "berlin"}):   8940,
    frozenset({"new york", "berlin"}):   6390,
}

_EXCHANGE_RATES: dict = {
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 149.50,
    "ILS": 3.70,
    "AUD": 1.53,
    "CAD": 1.36,
    "CHF": 0.89,
    "CNY": 7.24,
    "INR": 83.10,
    "USD": 1.0,
}


@tool
def calculate_trip_cost(
    flight_price: float,
    hotel_price_per_night: float,
    duration_days: int,
    activities_total: Optional[float] = None,
) -> str:
    """
    Calculate the total estimated cost of a trip.
    flight_price: round-trip flight cost in USD (must be >= 0).
    hotel_price_per_night: hotel cost per night in USD (must be >= 0).
    duration_days: number of nights (must be > 0).
    activities_total: optional total cost for activities in USD (must be >= 0).
    Returns a detailed cost breakdown including totals.
    """
    try:
        flight_price = float(flight_price)
        hotel_price_per_night = float(hotel_price_per_night)
        duration_days = int(duration_days)

        if flight_price < 0:
            return "Error: flight_price must be non-negative."
        if hotel_price_per_night < 0:
            return "Error: hotel_price_per_night must be non-negative."
        if duration_days <= 0:
            return "Error: duration_days must be greater than 0."

        hotel_total = hotel_price_per_night * duration_days
        grand_total = flight_price + hotel_total

        breakdown: dict = {
            "flight": f"${flight_price:.2f}",
            "hotel": f"${hotel_total:.2f} ({duration_days} nights × ${hotel_price_per_night:.2f})",
        }

        if activities_total is not None:
            activities_total = float(activities_total)
            if activities_total < 0:
                return "Error: activities_total must be non-negative."
            grand_total += activities_total
            breakdown["activities"] = f"${activities_total:.2f}"

        breakdown["total_estimate"] = f"${grand_total:.2f}"
        breakdown["currency"] = "USD"

        return json.dumps(breakdown, indent=2)

    except (ValueError, TypeError) as e:
        return f"Error: invalid input — {e}"


@tool
def currency_conversion(amount_usd: float, target_currency: str) -> str:
    """
    Convert a USD amount to another currency using fixed exchange rates.
    amount_usd: amount in US dollars (must be >= 0).
    target_currency: 3-letter currency code, e.g. EUR, GBP, JPY, ILS, AUD, CAD, CHF, CNY, INR.
    Returns the converted amount and the exchange rate used.
    """
    try:
        amount_usd = float(amount_usd)
        if amount_usd < 0:
            return "Error: amount_usd must be non-negative."

        code = target_currency.strip().upper()
        rate = _EXCHANGE_RATES.get(code)

        if rate is None:
            supported = ", ".join(sorted(_EXCHANGE_RATES.keys() - {"USD"}))
            return f"Error: unsupported currency '{code}'. Supported: {supported}."

        converted = amount_usd * rate

        return json.dumps({
            "original": f"${amount_usd:.2f} USD",
            "converted": f"{converted:.2f} {code}",
            "rate": f"1 USD = {rate} {code}",
            "note": "Rates are fixed approximations for trip planning purposes.",
        }, indent=2)

    except (ValueError, TypeError) as e:
        return f"Error: invalid input — {e}"


@tool
def estimate_daily_budget(
    total_budget: float,
    flight_price: float,
    duration_days: int,
    activities_total: Optional[float] = None,
) -> str:
    """
    Calculate how much money is left per day after fixed trip costs.
    total_budget: total available budget in USD.
    flight_price: round-trip flight cost in USD.
    duration_days: number of days (must be > 0).
    activities_total: optional pre-planned activities cost in USD.
    Returns daily budget remaining for hotels, food, and spending.
    """
    try:
        total_budget = float(total_budget)
        flight_price = float(flight_price)
        duration_days = int(duration_days)

        if total_budget < 0:
            return "Error: total_budget must be non-negative."
        if flight_price < 0:
            return "Error: flight_price must be non-negative."
        if duration_days <= 0:
            return "Error: duration_days must be greater than 0."

        remaining = total_budget - flight_price

        if activities_total is not None:
            activities_total = float(activities_total)
            if activities_total < 0:
                return "Error: activities_total must be non-negative."
            remaining -= activities_total

        if remaining < 0:
            return json.dumps({
                "warning": "Fixed costs exceed total budget.",
                "total_budget": f"${total_budget:.2f}",
                "fixed_costs": f"${total_budget - remaining:.2f}",
                "shortfall": f"${abs(remaining):.2f}",
            }, indent=2)

        daily = remaining / duration_days

        return json.dumps({
            "total_budget": f"${total_budget:.2f}",
            "flight_cost": f"${flight_price:.2f}",
            "activities_cost": f"${activities_total:.2f}" if activities_total else "not included",
            "remaining_after_fixed_costs": f"${remaining:.2f}",
            "duration_days": duration_days,
            "daily_budget": f"${daily:.2f}/day",
            "note": "Daily budget covers hotel, food, and personal spending.",
        }, indent=2)

    except (ValueError, TypeError) as e:
        return f"Error: invalid input — {e}"


@tool
def summarize_trip(
    destination_city: str,
    duration_days: int,
    total_cost_usd: float,
    visa_status: str,
    weather_description: Optional[str] = None,
    daily_budget_usd: Optional[float] = None,
) -> str:
    """
    Generate a one-line trip summary combining cost, visa, weather, and budget.
    destination_city: name of the destination city.
    duration_days: length of the trip in days (must be > 0).
    total_cost_usd: total estimated trip cost in USD (must be >= 0).
    visa_status: short visa status string (e.g. 'visa-free', 'ESTA required').
    weather_description: optional short weather note (e.g. '22C, sunny').
    daily_budget_usd: optional remaining daily budget in USD.
    Returns a concise human-readable trip summary.
    """
    try:
        duration_days = int(duration_days)
        total_cost_usd = float(total_cost_usd)

        if duration_days <= 0:
            return "Error: duration_days must be greater than 0."
        if total_cost_usd < 0:
            return "Error: total_cost_usd must be non-negative."

        parts = [
            f"{duration_days} days in {destination_city}",
            f"${total_cost_usd:.0f} total",
        ]

        if daily_budget_usd is not None:
            daily_budget_usd = float(daily_budget_usd)
            parts.append(f"${daily_budget_usd:.0f}/day spending")

        parts.append(visa_status)

        if weather_description:
            parts.append(weather_description)

        return json.dumps({"summary": " · ".join(parts)}, indent=2)

    except (ValueError, TypeError) as e:
        return f"Error: invalid input — {e}"


@tool
def distance_travel_time(origin_city: str, destination_city: str) -> str:
    """
    Estimate flight distance and travel time between two supported cities.
    origin_city: departure city (e.g. 'Tel Aviv', 'Paris', 'London').
    destination_city: arrival city (e.g. 'Tokyo', 'New York', 'Berlin').
    Returns distance in km and estimated flight duration.
    Supported cities: Tel Aviv, Paris, London, Tokyo, New York, Berlin.
    """
    origin = origin_city.strip().lower()
    destination = destination_city.strip().lower()

    if origin == destination:
        return json.dumps({
            "origin": origin_city,
            "destination": destination_city,
            "distance_km": 0,
            "estimated_flight_time": "0h 00m",
            "note": "Origin and destination are the same city.",
        }, indent=2)

    key = frozenset({origin, destination})
    distance_km = _FLIGHT_DISTANCES_KM.get(key)

    if distance_km is None:
        return (
            f"Distance data not available for {origin_city} to {destination_city}. "
            "Supported cities: Tel Aviv, Paris, London, Tokyo, New York, Berlin."
        )

    # Average commercial flight speed ~900 km/h + 30 min taxi/climb
    raw_hours = distance_km / 900 + 0.5
    hours = int(raw_hours)
    minutes = round((raw_hours - hours) * 60)

    return json.dumps({
        "origin": origin_city,
        "destination": destination_city,
        "distance_km": distance_km,
        "estimated_flight_time": f"{hours}h {minutes:02d}m",
        "note": "Estimate based on ~900 km/h cruising speed. Actual times vary by airline and route.",
    }, indent=2)


# ── generate_daily_itinerary ──────────────────────────────────────────────────

_DEFAULT_MORNING = [
    "Explore the city centre and main landmark",
    "Visit the national museum",
    "Walking tour of the old town",
    "Day trip to a nearby attraction",
]
_DEFAULT_AFTERNOON = [
    "Lunch at a local restaurant, afternoon shopping",
    "Art gallery and café break",
    "Park walk and local market",
    "Boat tour / city tour bus",
]
_DEFAULT_EVENING = [
    "Dinner at a recommended local restaurant",
    "Evening show / theatre / concert",
    "Sunset viewpoint and night market",
    "Jazz bar / rooftop dining experience",
]


@tool
def generate_daily_itinerary(
    destination_city: str,
    duration_days: int,
    activities: str = "",
    daily_budget_usd: float = 0.0,
) -> str:
    """
    Build a structured day-by-day itinerary for a trip.

    destination_city:  Target destination (e.g. "Tokyo").
    duration_days:     Number of trip days (1-30).
    activities:        Comma-separated list of planned activities (optional).
    daily_budget_usd:  Per-day spending budget in USD (optional, 0 = unspecified).

    Returns a JSON array with one entry per day: morning, afternoon, evening, budget note.
    """
    try:
        duration_days = int(duration_days)
        daily_budget_usd = float(daily_budget_usd)
    except (ValueError, TypeError) as e:
        return f"Error: invalid input — {e}"

    if duration_days <= 0 or duration_days > 30:
        return "Error: duration_days must be between 1 and 30."

    # Parse user-provided activities into a queue
    activity_list = [a.strip() for a in activities.split(",") if a.strip()] if activities else []

    itinerary = []
    for day in range(1, duration_days + 1):
        idx = (day - 1) % 4
        morning   = activity_list.pop(0) if activity_list else _DEFAULT_MORNING[idx]
        afternoon = activity_list.pop(0) if activity_list else _DEFAULT_AFTERNOON[idx]
        evening   = _DEFAULT_EVENING[idx]

        day_entry: dict = {
            "day": day,
            "morning": morning,
            "afternoon": afternoon,
            "evening": evening,
        }

        if daily_budget_usd > 0:
            day_entry["budget_note"] = f"~${daily_budget_usd:.0f} available for food, transport, and entrance fees"

        if day == 1:
            day_entry["note"] = "Arrival day — check in, rest, light exploration."
        elif day == duration_days:
            day_entry["note"] = "Departure day — check out, last-minute shopping, head to airport."

        itinerary.append(day_entry)

    return json.dumps({
        "destination": destination_city,
        "duration_days": duration_days,
        "itinerary": itinerary,
    }, indent=2)
