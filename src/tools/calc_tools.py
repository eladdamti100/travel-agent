"""
Calculation tools — pure arithmetic, no database or API calls.
"""

import json

from langchain_core.tools import tool


@tool
def calculate_trip_cost(
    flight_price: float,
    hotel_price_per_night: float,
    duration_days: int,
) -> str:
    """
    Calculate the total estimated cost of a trip.
    flight_price: round-trip flight cost in USD.
    hotel_price_per_night: hotel cost per night in USD.
    duration_days: number of nights.
    Returns a detailed cost breakdown including totals.
    """
    try:
        hotel_total = float(hotel_price_per_night) * int(duration_days)
        grand_total = float(flight_price) + hotel_total

        breakdown = {
            "flight": f"${float(flight_price):.2f}",
            "hotel": f"${hotel_total:.2f} ({duration_days} nights × ${float(hotel_price_per_night):.2f})",
            "total_estimate": f"${grand_total:.2f}",
            "currency": "USD",
        }

        return json.dumps(breakdown, indent=2)

    except (ValueError, TypeError) as e:
        return f"Error: invalid input — {e}"
