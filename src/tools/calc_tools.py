"""
Calculation tools — pure arithmetic, no database or API calls.
"""

import json
from typing import Optional

from langchain_core.tools import tool


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
