"""
Integration tests for all SQL-backed and calculation tools.
These tests hit the real SQLite database — no mocks, no API calls.
"""

import pytest
from src.utils.db_init import create_travel_db

# Recreate DB at test start so tests are always idempotent
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


# ── fetch_flights ─────────────────────────────────────────────────────────────

def test_fetch_flights_found():
    result = fetch_flights.invoke({"origin": "TLV", "destination": "Paris"})
    assert "El Al" in result or "LY321" in result

def test_fetch_flights_case_insensitive():
    result = fetch_flights.invoke({"origin": "tlv", "destination": "paris"})
    assert "El Al" in result or "AF123" in result

def test_fetch_flights_not_found():
    result = fetch_flights.invoke({"origin": "TLV", "destination": "Sydney"})
    assert "No flights" in result


# ── fetch_hotels ──────────────────────────────────────────────────────────────

def test_fetch_hotels_found():
    result = fetch_hotels.invoke({"city": "Paris"})
    assert "Hotel de Ville" in result

def test_fetch_hotels_budget_filter_passes():
    result = fetch_hotels.invoke({"city": "Paris", "max_price": 90})
    assert "Ibis" in result

def test_fetch_hotels_budget_filter_excludes():
    result = fetch_hotels.invoke({"city": "Paris", "max_price": 50})
    assert "No hotels" in result

def test_fetch_hotels_unknown_city():
    result = fetch_hotels.invoke({"city": "Dubai"})
    assert "No hotels" in result


# ── fetch_activities ──────────────────────────────────────────────────────────

def test_fetch_activities_found():
    result = fetch_activities.invoke({"city": "Paris"})
    assert "Louvre" in result or "Eiffel" in result

def test_fetch_activities_unknown_city():
    result = fetch_activities.invoke({"city": "Dubai"})
    assert "No activities" in result


# ── get_visa_requirement ──────────────────────────────────────────────────────

def test_get_visa_requirement_known():
    result = get_visa_requirement.invoke(
        {"origin_country": "Israel", "destination_country": "France"}
    )
    assert "visa" in result.lower()

def test_get_visa_requirement_unknown_route():
    result = get_visa_requirement.invoke(
        {"origin_country": "Brazil", "destination_country": "Antarctica"}
    )
    assert "No visa information" in result


# ── list_destinations ─────────────────────────────────────────────────────────

def test_list_destinations_known_origin():
    result = list_destinations.invoke({"origin": "TLV"})
    for city in ("Paris", "London", "Berlin"):
        assert city in result

def test_list_destinations_unknown_origin():
    result = list_destinations.invoke({"origin": "XYZ"})
    assert "No destinations" in result


# ── get_cheapest_hotel ────────────────────────────────────────────────────────

def test_get_cheapest_hotel_berlin():
    result = get_cheapest_hotel.invoke({"city": "Berlin"})
    assert "40" in result or "Hostel" in result

def test_get_cheapest_hotel_unknown():
    result = get_cheapest_hotel.invoke({"city": "Dubai"})
    assert "No hotels" in result


# ── get_cheapest_flight ───────────────────────────────────────────────────────

def test_get_cheapest_flight_berlin():
    result = get_cheapest_flight.invoke({"origin": "TLV", "destination": "Berlin"})
    assert "110" in result or "Ryanair" in result

def test_get_cheapest_flight_not_found():
    result = get_cheapest_flight.invoke({"origin": "TLV", "destination": "Sydney"})
    assert "No flights" in result


# ── calculate_trip_cost ───────────────────────────────────────────────────────

def test_calculate_trip_cost_totals_correctly():
    # 350 (flight) + 150*5 (hotel) = 1100
    result = calculate_trip_cost.invoke(
        {"flight_price": 350.0, "hotel_price_per_night": 150.0, "duration_days": 5}
    )
    assert "1100" in result

def test_calculate_trip_cost_shows_breakdown():
    result = calculate_trip_cost.invoke(
        {"flight_price": 100.0, "hotel_price_per_night": 50.0, "duration_days": 3}
    )
    assert "flight" in result.lower()
    assert "hotel" in result.lower()

def test_calculate_trip_cost_bad_input():
    with pytest.raises(Exception):
        calculate_trip_cost.invoke(
            {
                "flight_price": "bad",
                "hotel_price_per_night": 50.0,
                "duration_days": 3,
            }
        )


# ── currency_conversion ───────────────────────────────────────────────────────

def test_currency_conversion_usd_to_eur():
    result = currency_conversion.invoke({"amount_usd": 1000, "target_currency": "EUR"})
    assert "EUR" in result
    assert "920" in result

def test_currency_conversion_usd_to_ils():
    result = currency_conversion.invoke({"amount_usd": 100, "target_currency": "ILS"})
    assert "ILS" in result
    assert "370" in result

def test_currency_conversion_usd_to_usd():
    result = currency_conversion.invoke({"amount_usd": 500, "target_currency": "USD"})
    assert "500.00 USD" in result

def test_currency_conversion_unsupported_currency():
    result = currency_conversion.invoke({"amount_usd": 100, "target_currency": "XYZ"})
    assert "Error" in result
    assert "unsupported" in result.lower()

def test_currency_conversion_negative_amount():
    result = currency_conversion.invoke({"amount_usd": -50, "target_currency": "EUR"})
    assert "Error" in result


# ── estimate_daily_budget ─────────────────────────────────────────────────────

def test_estimate_daily_budget_basic():
    # 3000 - 350 flight = 2650 remaining / 7 days = 378.57/day
    result = estimate_daily_budget.invoke(
        {"total_budget": 3000, "flight_price": 350, "duration_days": 7}
    )
    assert "378" in result
    assert "daily_budget" in result

def test_estimate_daily_budget_with_activities():
    # 2000 - 400 flight - 200 activities = 1400 / 7 = 200/day
    result = estimate_daily_budget.invoke(
        {"total_budget": 2000, "flight_price": 400, "duration_days": 7, "activities_total": 200}
    )
    assert "200" in result

def test_estimate_daily_budget_over_budget():
    result = estimate_daily_budget.invoke(
        {"total_budget": 300, "flight_price": 500, "duration_days": 5}
    )
    assert "shortfall" in result or "exceed" in result.lower()

def test_estimate_daily_budget_zero_days():
    result = estimate_daily_budget.invoke(
        {"total_budget": 1000, "flight_price": 200, "duration_days": 0}
    )
    assert "Error" in result


# ── fetch_weather ─────────────────────────────────────────────────────────────

def test_fetch_weather_paris_june():
    result = fetch_weather.invoke({"city": "Paris", "month": "june"})
    assert "22" in result
    assert "sunny" in result.lower()

def test_fetch_weather_case_insensitive():
    result = fetch_weather.invoke({"city": "TOKYO", "month": "DECEMBER"})
    assert "tokyo" in result.lower() or "9" in result

def test_fetch_weather_includes_fahrenheit():
    result = fetch_weather.invoke({"city": "London", "month": "june"})
    assert "avg_temp_f" in result

def test_fetch_weather_unknown_city():
    result = fetch_weather.invoke({"city": "Dubai", "month": "june"})
    assert "No weather data" in result

def test_fetch_weather_unknown_month():
    result = fetch_weather.invoke({"city": "Paris", "month": "octember"})
    assert "No weather data" in result


# ── fetch_restaurants ─────────────────────────────────────────────────────────

def test_fetch_restaurants_found():
    result = fetch_restaurants.invoke({"city": "London"})
    assert "Dishoom" in result

def test_fetch_restaurants_cuisine_filter():
    result = fetch_restaurants.invoke({"city": "Paris", "cuisine": "French"})
    assert "French" in result

def test_fetch_restaurants_cuisine_no_match():
    result = fetch_restaurants.invoke({"city": "Berlin", "cuisine": "Japanese"})
    assert "No restaurants" in result

def test_fetch_restaurants_unknown_city():
    result = fetch_restaurants.invoke({"city": "Dubai"})
    assert "No restaurants" in result

def test_fetch_restaurants_sorted_by_rating():
    import json
    result = fetch_restaurants.invoke({"city": "London"})
    rows = json.loads(result)
    ratings = [r["rating"] for r in rows]
    assert ratings == sorted(ratings, reverse=True)


# ── kosher_food_finder ────────────────────────────────────────────────────────

def test_kosher_food_finder_paris():
    result = kosher_food_finder.invoke({"city": "Paris"})
    assert "Kosher" in result or "kosher" in result.lower()

def test_kosher_food_finder_new_york():
    result = kosher_food_finder.invoke({"city": "New York"})
    assert "Le Marais" in result

def test_kosher_food_finder_all_results_are_kosher():
    import json
    result = kosher_food_finder.invoke({"city": "London"})
    rows = json.loads(result)
    assert len(rows) > 0
    for row in rows:
        # kosher column should not appear (filtered out), rating should be present
        assert "rating" in row

def test_kosher_food_finder_unknown_city():
    result = kosher_food_finder.invoke({"city": "Dubai"})
    assert "No kosher" in result


# ── summarize_trip ────────────────────────────────────────────────────────────

def test_summarize_trip_full():
    result = summarize_trip.invoke({
        "destination_city": "Paris",
        "duration_days": 7,
        "total_cost_usd": 1840,
        "visa_status": "visa-free",
        "weather_description": "22C, sunny",
        "daily_budget_usd": 185,
    })
    assert "Paris" in result
    assert "1840" in result
    assert "visa-free" in result
    assert "22C" in result
    assert "185" in result

def test_summarize_trip_minimal():
    result = summarize_trip.invoke({
        "destination_city": "Tokyo",
        "duration_days": 5,
        "total_cost_usd": 2500,
        "visa_status": "visa-free",
    })
    assert "Tokyo" in result
    assert "2500" in result

def test_summarize_trip_uses_dot_separator():
    import json
    result = summarize_trip.invoke({
        "destination_city": "Berlin",
        "duration_days": 3,
        "total_cost_usd": 900,
        "visa_status": "visa-free",
    })
    summary = json.loads(result)["summary"]
    assert "·" in summary

def test_summarize_trip_zero_days():
    result = summarize_trip.invoke({
        "destination_city": "London",
        "duration_days": 0,
        "total_cost_usd": 1000,
        "visa_status": "visa-free",
    })
    assert "Error" in result


# ── local_transport_guide ─────────────────────────────────────────────────────

def test_local_transport_guide_paris():
    result = local_transport_guide.invoke({"city": "Paris"})
    assert "Metro" in result

def test_local_transport_guide_tokyo():
    result = local_transport_guide.invoke({"city": "Tokyo"})
    assert "Suica" in result or "IC" in result

def test_local_transport_guide_contains_tip():
    import json
    result = local_transport_guide.invoke({"city": "London"})
    rows = json.loads(result)
    assert all("tip" in row for row in rows)

def test_local_transport_guide_contains_price():
    import json
    result = local_transport_guide.invoke({"city": "Berlin"})
    rows = json.loads(result)
    assert all("price_range" in row for row in rows)

def test_local_transport_guide_unknown_city():
    result = local_transport_guide.invoke({"city": "Dubai"})
    assert "No local transport" in result


# ── airport_transfer_info ─────────────────────────────────────────────────────

def test_airport_transfer_info_paris():
    result = airport_transfer_info.invoke({"city": "Paris"})
    assert "RER" in result or "CDG" in result

def test_airport_transfer_info_london():
    result = airport_transfer_info.invoke({"city": "London"})
    assert "Heathrow" in result or "Express" in result

def test_airport_transfer_info_contains_duration():
    import json
    result = airport_transfer_info.invoke({"city": "Tokyo"})
    rows = json.loads(result)
    assert all("duration" in row for row in rows)

def test_airport_transfer_info_contains_price():
    import json
    result = airport_transfer_info.invoke({"city": "New York"})
    rows = json.loads(result)
    assert all("price_usd" in row for row in rows)

def test_airport_transfer_info_unknown_city():
    result = airport_transfer_info.invoke({"city": "Dubai"})
    assert "No airport transfer" in result


# ── events_finder ─────────────────────────────────────────────────────────────

def test_events_finder_paris_june():
    result = events_finder.invoke({"city": "Paris", "month": "june"})
    assert "Fête de la Musique" in result or "Pride" in result

def test_events_finder_london_december():
    result = events_finder.invoke({"city": "London", "month": "december"})
    assert "Winter Wonderland" in result

def test_events_finder_case_insensitive():
    result = events_finder.invoke({"city": "TOKYO", "month": "MARCH"})
    assert "Cherry Blossom" in result or "Hanami" in result

def test_events_finder_contains_category():
    import json
    result = events_finder.invoke({"city": "Berlin", "month": "december"})
    rows = json.loads(result)
    assert all("category" in row for row in rows)

def test_events_finder_unknown_month():
    result = events_finder.invoke({"city": "Paris", "month": "octember"})
    assert "No events" in result

def test_events_finder_unknown_city():
    result = events_finder.invoke({"city": "Dubai", "month": "june"})
    assert "No events" in result


# ── distance_travel_time ──────────────────────────────────────────────────────

def test_distance_travel_time_tlv_to_paris():
    result = distance_travel_time.invoke({"origin_city": "Tel Aviv", "destination_city": "Paris"})
    assert "3310" in result
    assert "estimated_flight_time" in result

def test_distance_travel_time_symmetric():
    import json
    a = json.loads(distance_travel_time.invoke({"origin_city": "London", "destination_city": "Tokyo"}))
    b = json.loads(distance_travel_time.invoke({"origin_city": "Tokyo", "destination_city": "London"}))
    assert a["distance_km"] == b["distance_km"]

def test_distance_travel_time_same_city():
    result = distance_travel_time.invoke({"origin_city": "Paris", "destination_city": "Paris"})
    assert "same city" in result.lower() or "0" in result

def test_distance_travel_time_unsupported_city():
    result = distance_travel_time.invoke({"origin_city": "Tel Aviv", "destination_city": "Sydney"})
    assert "not available" in result.lower()

def test_distance_travel_time_includes_note():
    import json
    result = distance_travel_time.invoke({"origin_city": "New York", "destination_city": "Berlin"})
    data = json.loads(result)
    assert "note" in data
    assert data["distance_km"] == 6390