"""
Tests for context_enricher — extract_trip_context_deterministic and its sub-extractors.
Pure regex/state functions, no LLM calls, no mocking needed.
"""

import pytest
from langchain_core.messages import HumanMessage


def _state(text, **overrides):
    base = {"messages": [HumanMessage(content=text)]}
    base.update(overrides)
    return base


@pytest.mark.parametrize("fn_name,text,expected", [
    ("_extract_destination_city",  "i want to visit paris",       "Paris"),
    ("_extract_destination_city",  "trip to tokyo",               "Tokyo"),
    ("_extract_destination_city",  "going to new york city",      "New York"),
    ("_extract_destination_city",  "i like cats",                 None),
    ("_extract_origin_airport",    "flying from TLV to Paris",    "TLV"),
    ("_extract_origin_airport",    "depart from JFK next week",   "JFK"),
    ("_extract_origin_airport",    "plan a trip to Paris",        None),
    ("_extract_origin_country",    "i am from israel",            "Israel"),
    ("_extract_origin_country",    "i'm from the usa",            "United States"),
    ("_extract_origin_country",    "passport is uk",              "United Kingdom"),
    ("_extract_origin_country",    "plan a trip to Paris",        None),
    ("_extract_duration_days",     "a 5-day trip",                5),
    ("_extract_duration_days",     "for 7 days",                  7),
    ("_extract_duration_days",     "staying 3 nights",            3),
    ("_extract_duration_days",     "trip to paris",               None),
    ("_extract_total_budget",      "budget of $2000",             2000.0),
    ("_extract_total_budget",      "up to $1,500",                1500.0),
    ("_extract_total_budget",      "3000 usd",                    3000.0),
    ("_extract_total_budget",      "plan a trip to paris",        None),
    ("_extract_num_travelers",     "we are 4 people",             4),
    ("_extract_num_travelers",     "family of 3",                 3),
    ("_extract_num_travelers",     "just me going to paris",      None),
])
def test_individual_extractors(fn_name, text, expected):
    import src.agents.context_enricher as mod
    result = getattr(mod, fn_name)(text)
    assert result == expected


def test_preference_extractors():
    from src.agents.context_enricher import (
        _extract_hotel_preference, _extract_flight_preference,
        _extract_activity_preference, _extract_travel_style,
    )
    assert _extract_hotel_preference("i want a 5-star hotel") == "Prefers 5-star hotels"
    assert _extract_hotel_preference("budget hotel please") == "Prefers budget hotels"
    assert _extract_flight_preference("direct flight only") == "Direct flights only"
    assert _extract_flight_preference("business class preferred") == "Prefers business class"
    assert _extract_activity_preference("i love museums") == "Prefers museums"
    assert _extract_activity_preference("trip with kids") == "Prefers family-friendly activities"
    assert _extract_travel_style("luxury trip") == "luxury"
    assert _extract_travel_style("budget travel") == "budget"


def test_full_and_fallback_extraction():
    from src.agents.context_enricher import extract_trip_context_deterministic

    ctx = extract_trip_context_deterministic(
        _state("I am from Israel, flying from TLV, planning a 5-day trip to Paris with a budget of $2000 for 2 people")
    )
    assert ctx.destination_city == "Paris"
    assert ctx.destination_country == "France"
    assert ctx.origin_airport == "TLV"
    assert ctx.duration_days == 5
    assert ctx.total_budget == 2000.0
    assert ctx.num_travelers == 2
    assert ctx.extraction_source == "deterministic"

    ctx = extract_trip_context_deterministic(_state("I want to go there", current_city="Tokyo", total_budget=3000.0))
    assert ctx.destination_city == "Tokyo" and ctx.total_budget == 3000.0

    ctx = extract_trip_context_deterministic(_state(""))
    assert ctx.destination_city is None and ctx.total_budget is None and ctx.origin_airport is None
