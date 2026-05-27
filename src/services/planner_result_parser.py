"""
Parsers for converting raw planner tool output into typed planner results.
"""

import json
from typing import Dict, List, Optional

from src.models.planner import (
    ActivityResult,
    CostResult,
    FlightResult,
    HotelResult,
    PlannerTaskType,
    PlannerToolResults,
    VisaResult,
)
from src.models.trip_context import TripContext

def clean_money_value(value) -> Optional[float]:
    """
    Converts money-like values such as "$1,200.00" into float.
    """
    if value in (None, ""):
        return None

    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()

    try:
        return float(value)
    except (TypeError, ValueError):
        return None
    
def extract_lowest_price_from_json(
    raw_json: str,
    *,
    price_key: str = "price",
) -> Optional[float]:
    """
    Extracts the lowest price from a tool JSON response.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(data, dict):
        value = data.get(price_key)
        return float(value) if value is not None else None

    if not isinstance(data, list):
        return None

    prices = []

    for item in data:
        if not isinstance(item, dict):
            continue

        value = item.get(price_key)
        if value is None:
            continue

        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            continue

    return min(prices) if prices else None


def build_structured_tool_results(
    *,
    context: TripContext,
    raw_results: Dict[str, str],
) -> PlannerToolResults:
    """
    Builds a structured planner output container from existing raw tool results.
    """
    return PlannerToolResults(
        flights=parse_flight_results(
            raw_results.get(PlannerTaskType.FETCH_FLIGHTS.value, "")
        ),
        hotels=parse_hotel_results(
            raw_results.get(PlannerTaskType.FETCH_HOTELS.value, "")
        ),
        activities=parse_activity_results(
            raw_results.get(PlannerTaskType.FETCH_ACTIVITIES.value, "")
        ),
        visa=parse_visa_result(
            raw_results.get(PlannerTaskType.CHECK_VISA.value, ""),
            context=context,
        ),
        cost=parse_cost_result(
            raw_results.get(PlannerTaskType.CALCULATE_TRIP_COST.value, ""),
            context=context,
        ),
        raw_results=raw_results,
    )


def parse_json_result(raw: str):
    """
    Safely parses a JSON-like tool result.

    Returns None when the tool output is not valid JSON.
    """
    if not raw:
        return None

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def ensure_list(data) -> list:
    """
    Normalizes parsed tool data into a list.
    """
    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data]

    return []


def as_float(value) -> Optional[float]:
    """
    Converts a value to float when possible.
    """
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value) -> Optional[int]:
    """
    Converts a value to int when possible.
    """
    if value in (None, ""):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_flight_results(raw: str) -> List[FlightResult]:
    """
    Parses raw flight tool output into structured FlightResult objects.
    """
    rows = ensure_list(parse_json_result(raw))
    results: List[FlightResult] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        results.append(
            FlightResult(
                origin=row.get("origin"),
                destination=row.get("destination"),
                airline=row.get("airline"),
                flight_number=row.get("flight_number") or row.get("flight"),
                price=as_float(row.get("price")),
                departure_time=row.get("departure_time"),
                arrival_time=row.get("arrival_time"),
                duration=row.get("duration"),
                raw=row,
            )
        )

    return results


def parse_hotel_results(raw: str) -> List[HotelResult]:
    """
    Parses raw hotel tool output into structured HotelResult objects.
    """
    rows = ensure_list(parse_json_result(raw))
    results: List[HotelResult] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        results.append(
            HotelResult(
                city=row.get("city"),
                name=row.get("name") or row.get("hotel"),
                price_per_night=as_float(row.get("price_per_night")),
                rating=as_float(row.get("rating")),
                location=row.get("location"),
                raw=row,
            )
        )

    return results


def parse_activity_results(raw: str) -> List[ActivityResult]:
    """
    Parses raw activity tool output into structured ActivityResult objects.
    """
    rows = ensure_list(parse_json_result(raw))
    results: List[ActivityResult] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        results.append(
            ActivityResult(
                city=row.get("city"),
                name=row.get("name") or row.get("activity"),
                category=row.get("category"),
                price=as_float(row.get("price")),
                duration=row.get("duration"),
                raw=row,
            )
        )

    return results


def parse_visa_result(
    raw: str,
    *,
    context: TripContext,
) -> Optional[VisaResult]:
    """
    Parses raw visa tool output into a structured VisaResult.
    """
    data = parse_json_result(raw)

    if not isinstance(data, dict):
        if not raw:
            return None

        return VisaResult(
            origin_country=context.origin_country,
            destination_country=context.destination_country,
            requirement_summary=raw,
            raw={},
        )

    return VisaResult(
        origin_country=data.get("origin_country") or context.origin_country,
        destination_country=(
            data.get("destination_country") or context.destination_country
        ),
        visa_required=data.get("visa_required"),
        requirement_summary=(
            data.get("requirement_summary")
            or data.get("requirement")
            or data.get("summary")
        ),
        raw=data,
    )


def parse_cost_result(
    raw: str,
    *,
    context: TripContext,
) -> Optional[CostResult]:
    """
    Parses raw cost tool output into a structured CostResult.
    """
    data = parse_json_result(raw)

    if not isinstance(data, dict):
        return None

    total_cost = (
        clean_money_value(data.get("total_cost"))
        or clean_money_value(data.get("total"))
        or clean_money_value(data.get("estimated_total"))
        or clean_money_value(data.get("total_estimate"))
    )

    within_budget = None
    if total_cost is not None and context.total_budget is not None:
        within_budget = total_cost <= context.total_budget

    return CostResult(
        flight_price=clean_money_value(data.get("flight_price")),
        hotel_price_per_night=clean_money_value(data.get("hotel_price_per_night")),
        duration_days=as_int(data.get("duration_days")) or context.duration_days,
        total_cost=total_cost,
        within_budget=within_budget,
        raw=data,
    )
