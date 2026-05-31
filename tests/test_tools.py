"""
Integration tests for all SQL-backed and calculation tools.
These tests hit the real SQLite database — no mocks, no API calls.
"""

import json
import pytest
from src.utils.db_init import create_travel_db

create_travel_db()

from src.tools.db_tools import (
    airport_transfer_info,
    events_finder,
    fetch_activities,
    fetch_flights,
    fetch_hotels,
    fetch_restaurants,
    fetch_weather,
    get_cheapest_flight,
    get_cheapest_hotel,
    get_visa_requirement,
    kosher_food_finder,
    list_destinations,
    local_transport_guide,
)
from src.tools.calc_tools import (
    calculate_trip_cost,
    currency_conversion,
    distance_travel_time,
    estimate_daily_budget,
    summarize_trip,
)


@pytest.mark.parametrize("fn,args,expected", [
    # fetch_flights
    (fetch_flights, {"origin": "TLV", "destination": "Paris"},  "El Al"),
    (fetch_flights, {"origin": "tlv", "destination": "paris"},  "El Al"),
    (fetch_flights, {"origin": "TLV", "destination": "Sydney"}, "No flights"),
    # fetch_hotels
    (fetch_hotels, {"city": "Paris"},                           "Hotel de Ville"),
    (fetch_hotels, {"city": "Paris", "max_price": 90},          "Ibis"),
    (fetch_hotels, {"city": "Paris", "max_price": 50},          "No hotels"),
    (fetch_hotels, {"city": "Dubai"},                           "No hotels"),
    # fetch_activities
    (fetch_activities, {"city": "Paris"},                       "Louvre"),
    (fetch_activities, {"city": "Dubai"},                       "No activities"),
    # visa
    (get_visa_requirement, {"origin_country": "Israel",  "destination_country": "France"},     "visa"),
    (get_visa_requirement, {"origin_country": "Brazil",  "destination_country": "Antarctica"}, "No visa information"),
    # list_destinations
    (list_destinations, {"origin": "TLV"}, "Paris"),
    (list_destinations, {"origin": "XYZ"}, "No destinations"),
    # cheapest
    (get_cheapest_hotel,  {"city": "Berlin"},                          "40"),
    (get_cheapest_hotel,  {"city": "Dubai"},                           "No hotels"),
    (get_cheapest_flight, {"origin": "TLV", "destination": "Berlin"},  "110"),
    (get_cheapest_flight, {"origin": "TLV", "destination": "Sydney"},  "No flights"),
    # weather
    (fetch_weather, {"city": "Paris",  "month": "june"},      "22"),
    (fetch_weather, {"city": "London", "month": "june"},      "avg_temp_f"),
    (fetch_weather, {"city": "Dubai",  "month": "june"},      "No weather data"),
    (fetch_weather, {"city": "Paris",  "month": "octember"},  "No weather data"),
    # restaurants
    (fetch_restaurants, {"city": "London"},                          "Dishoom"),
    (fetch_restaurants, {"city": "Paris",  "cuisine": "French"},     "French"),
    (fetch_restaurants, {"city": "Berlin", "cuisine": "Japanese"},   "No restaurants"),
    (fetch_restaurants, {"city": "Dubai"},                           "No restaurants"),
    # kosher
    (kosher_food_finder, {"city": "New York"}, "Le Marais"),
    (kosher_food_finder, {"city": "Dubai"},    "No kosher"),
    # transport
    (local_transport_guide, {"city": "Paris"},  "Metro"),
    (local_transport_guide, {"city": "Tokyo"},  "Suica"),
    (local_transport_guide, {"city": "Dubai"},  "No local transport"),
    (airport_transfer_info, {"city": "Paris"},  "RER"),
    (airport_transfer_info, {"city": "London"}, "Heathrow"),
    (airport_transfer_info, {"city": "Dubai"},  "No airport transfer"),
    # events
    (events_finder, {"city": "London", "month": "december"}, "Winter Wonderland"),
    (events_finder, {"city": "TOKYO",  "month": "MARCH"},    "Cherry Blossom"),
    (events_finder, {"city": "Paris",  "month": "octember"}, "No events"),
    (events_finder, {"city": "Dubai",  "month": "june"},     "No events"),
])
def test_db_tool_lookup(fn, args, expected):
    result = fn.invoke(args)
    assert expected.lower() in result.lower() or expected in result


def test_json_row_fields():
    assert all("tip" in r        for r in json.loads(local_transport_guide.invoke({"city": "London"})))
    assert all("price_range" in r for r in json.loads(local_transport_guide.invoke({"city": "Berlin"})))
    assert all("duration" in r   for r in json.loads(airport_transfer_info.invoke({"city": "Tokyo"})))
    assert all("price_usd" in r  for r in json.loads(airport_transfer_info.invoke({"city": "New York"})))
    assert all("category" in r   for r in json.loads(events_finder.invoke({"city": "Berlin", "month": "december"})))
    rows = json.loads(kosher_food_finder.invoke({"city": "London"}))
    assert len(rows) > 0 and all("rating" in r for r in rows)


def test_restaurants_sorted_by_rating():
    rows = json.loads(fetch_restaurants.invoke({"city": "London"}))
    ratings = [r["rating"] for r in rows]
    assert ratings == sorted(ratings, reverse=True)


def test_calculate_trip_cost():
    result = calculate_trip_cost.invoke({"flight_price": 350.0, "hotel_price_per_night": 150.0, "duration_days": 5})
    assert "1100" in result and "flight" in result.lower() and "hotel" in result.lower()
    with pytest.raises(Exception):
        calculate_trip_cost.invoke({"flight_price": "bad", "hotel_price_per_night": 50.0, "duration_days": 3})


@pytest.mark.parametrize("amount,currency,expected", [
    (1000, "EUR", "920"),
    (100,  "ILS", "370"),
    (500,  "USD", "500.00 USD"),
    (100,  "XYZ", "unsupported"),
    (-50,  "EUR", "Error"),
])
def test_currency_conversion(amount, currency, expected):
    result = currency_conversion.invoke({"amount_usd": amount, "target_currency": currency})
    assert expected.lower() in result.lower() or expected in result


@pytest.mark.parametrize("kwargs,expected", [
    ({"total_budget": 3000, "flight_price": 350, "duration_days": 7},                          "378"),
    ({"total_budget": 2000, "flight_price": 400, "duration_days": 7, "activities_total": 200}, "200"),
    ({"total_budget": 300,  "flight_price": 500, "duration_days": 5},                          "shortfall"),
    ({"total_budget": 1000, "flight_price": 200, "duration_days": 0},                          "Error"),
])
def test_estimate_daily_budget(kwargs, expected):
    result = estimate_daily_budget.invoke(kwargs)
    assert expected.lower() in result.lower() or expected in result


def test_summarize_trip():
    full = summarize_trip.invoke({
        "destination_city": "Paris", "duration_days": 7, "total_cost_usd": 1840,
        "visa_status": "visa-free", "weather_description": "22C, sunny", "daily_budget_usd": 185,
    })
    assert all(s in full for s in ("Paris", "1840", "visa-free", "22C", "185"))
    assert "·" in json.loads(full)["summary"]

    assert "Tokyo" in summarize_trip.invoke({
        "destination_city": "Tokyo", "duration_days": 5, "total_cost_usd": 2500, "visa_status": "visa-free",
    })
    assert "Error" in summarize_trip.invoke({
        "destination_city": "London", "duration_days": 0, "total_cost_usd": 1000, "visa_status": "visa-free",
    })


def test_distance_travel_time():
    result = distance_travel_time.invoke({"origin_city": "Tel Aviv", "destination_city": "Paris"})
    assert "3310" in result and "estimated_flight_time" in result

    a = json.loads(distance_travel_time.invoke({"origin_city": "London", "destination_city": "Tokyo"}))
    b = json.loads(distance_travel_time.invoke({"origin_city": "Tokyo",  "destination_city": "London"}))
    assert a["distance_km"] == b["distance_km"]

    assert "not available" in distance_travel_time.invoke(
        {"origin_city": "Tel Aviv", "destination_city": "Sydney"}
    ).lower()
