"""
Database-backed travel tools.

All tools query the local SQLite travel database (data/travel_agency.db).
They are bound to the planner and researcher agents via ALL_TOOLS.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, Union

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent.parent.parent / "data" / "travel_agency.db"


def _run_query(query: str, params: tuple = ()) -> Union[list, str]:
    """Execute a parameterised SQL query and return rows as dicts, or an error string."""
    if not DB_PATH.exists():
        return "Error: Database not found. Run `python -m src.utils.db_init` first."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in rows]
    except sqlite3.Error as e:
        return f"Database error: {e}"


@tool
def fetch_flights(origin: str, destination: str) -> str:
    """
    Search for available flights between two locations.
    origin: 3-letter airport code (e.g. 'TLV', 'JFK', 'LHR').
    destination: full city name (e.g. 'Paris', 'London', 'Tokyo').
    Returns a list of flights with airline, price, and flight number.
    """
    query = """
        SELECT airline, price, flight_number
        FROM flights
        WHERE LOWER(origin) = ? AND LOWER(destination) = ?
        ORDER BY price ASC LIMIT 10
    """
    results = _run_query(query, (origin.strip().lower(), destination.strip().lower()))
    if isinstance(results, str):
        return results
    if not results:
        return f"No flights found from {origin} to {destination}."
    return json.dumps(results, indent=2)


@tool
def get_cheapest_flight(origin: str, destination: str) -> str:
    """
    Find the single cheapest flight between two locations.
    origin: 3-letter airport code. destination: city name.
    """
    query = """
        SELECT airline, price, flight_number
        FROM flights
        WHERE LOWER(origin) = ? AND LOWER(destination) = ?
        ORDER BY price ASC LIMIT 1
    """
    results = _run_query(query, (origin.strip().lower(), destination.strip().lower()))
    if isinstance(results, str):
        return results
    if not results:
        return f"No flights found from {origin} to {destination}."
    return json.dumps(results[0], indent=2)


@tool
def list_destinations(origin: str) -> str:
    """
    List all available flight destinations from a given origin airport.
    origin: 3-letter airport code (e.g. 'TLV').
    Returns a JSON array of destination city names.
    """
    query = "SELECT DISTINCT destination FROM flights WHERE LOWER(origin) = ? ORDER BY destination"
    results = _run_query(query, (origin.strip().lower(),))
    if isinstance(results, str):
        return results
    if not results:
        return f"No destinations found from {origin}."
    return json.dumps([r["destination"] for r in results], indent=2)


@tool
def fetch_hotels(city: str, max_price: Optional[float] = None) -> str:
    """
    Find hotels in a specific city, optionally filtered by max price per night.
    city: city name (e.g. 'Paris'). max_price: optional USD ceiling per night.
    Returns hotel name, stars, and price per night.
    """
    query = "SELECT name, price_per_night, stars FROM hotels WHERE LOWER(city) = ?"
    params: list = [city.strip().lower()]
    if max_price is not None:
        query += " AND price_per_night <= ?"
        params.append(max_price)
    query += " ORDER BY price_per_night ASC LIMIT 10"

    results = _run_query(query, tuple(params))
    if isinstance(results, str):
        return results
    if not results:
        suffix = f" under ${max_price}/night" if max_price else ""
        return f"No hotels found in {city}{suffix}."
    return json.dumps(results, indent=2)


@tool
def get_cheapest_hotel(city: str) -> str:
    """
    Find the single cheapest hotel in a given city.
    city: city name.
    """
    query = """
        SELECT name, price_per_night, stars
        FROM hotels
        WHERE LOWER(city) = ?
        ORDER BY price_per_night ASC LIMIT 1
    """
    results = _run_query(query, (city.strip().lower(),))
    if isinstance(results, str):
        return results
    if not results:
        return f"No hotels found in {city}."
    return json.dumps(results[0], indent=2)


@tool
def fetch_activities(city: str) -> str:
    """
    Find tourist activities and attractions in a city.
    city: city name. Returns activities with name, category, and price.
    """
    query = """
        SELECT name, category, price
        FROM activities
        WHERE LOWER(city) = ?
        ORDER BY price ASC LIMIT 10
    """
    results = _run_query(query, (city.strip().lower(),))
    if isinstance(results, str):
        return results
    if not results:
        return f"No activities found in {city}."
    return json.dumps(results, indent=2)


_COUNTRY_ALIASES_DB = {
    "united states": "usa",
    "united kingdom": "uk",
    "united arab emirates": "uae",
}


def _normalize_country_for_db(name: str) -> str:
    """Normalises a canonical country name to the short form stored in the DB."""
    key = name.strip().lower()
    return _COUNTRY_ALIASES_DB.get(key, key)


@tool
def get_visa_requirement(origin_country: str, destination_country: str) -> str:
    """
    Check visa entry requirements for travelers between two countries.
    origin_country: country of passport (e.g. 'Israel', 'India', 'USA').
    destination_country: destination country (e.g. 'France', 'Japan', 'UK').
    """
    query = """
        SELECT requirement
        FROM visa_requirements
        WHERE LOWER(origin_country) = ? AND LOWER(destination_country) = ?
    """
    results = _run_query(
        query,
        (
            _normalize_country_for_db(origin_country),
            _normalize_country_for_db(destination_country),
        ),
    )
    if isinstance(results, str):
        return results
    if not results:
        return (
            f"No visa information found for {origin_country} → {destination_country}. "
            "Please check the official embassy website."
        )
    return results[0]["requirement"]


@tool
def fetch_weather(city: str, month: str) -> str:
    """
    Get seasonal weather information for a city in a given month.
    city: city name (e.g. 'Paris', 'Tokyo').
    month: month name in English (e.g. 'june', 'december').
    Returns average temperature, description, and rainfall level.
    """
    query = """
        SELECT avg_temp_c, description, rainfall
        FROM weather
        WHERE LOWER(city) = ? AND LOWER(month) = ?
    """
    results = _run_query(query, (city.strip().lower(), month.strip().lower()))
    if isinstance(results, str):
        return results
    if not results:
        return f"No weather data found for {city} in {month}."
    row = results[0]
    return json.dumps({
        "city": city,
        "month": month,
        "avg_temp_c": row["avg_temp_c"],
        "avg_temp_f": round(row["avg_temp_c"] * 9 / 5 + 32),
        "description": row["description"],
        "rainfall": row["rainfall"],
    }, indent=2)


@tool
def fetch_restaurants(city: str, cuisine: Optional[str] = None) -> str:
    """
    Find restaurants in a city, optionally filtered by cuisine type.
    city: city name (e.g. 'Paris', 'New York').
    cuisine: optional cuisine filter (e.g. 'French', 'Sushi', 'Indian').
    Returns restaurant name, cuisine, price range, and rating.
    """
    query = "SELECT name, cuisine, price_range, kosher, rating FROM restaurants WHERE LOWER(city) = ?"
    params: list = [city.strip().lower()]
    if cuisine:
        query += " AND LOWER(cuisine) LIKE ?"
        params.append(f"%{cuisine.strip().lower()}%")
    query += " ORDER BY rating DESC LIMIT 10"

    results = _run_query(query, tuple(params))
    if isinstance(results, str):
        return results
    if not results:
        suffix = f" ({cuisine} cuisine)" if cuisine else ""
        return f"No restaurants found in {city}{suffix}."
    return json.dumps(results, indent=2)


@tool
def kosher_food_finder(city: str) -> str:
    """
    Find kosher-certified restaurants in a city.
    city: city name (e.g. 'Paris', 'London', 'New York').
    Returns kosher restaurants with name, cuisine, price range, and rating.
    """
    query = """
        SELECT name, cuisine, price_range, rating
        FROM restaurants
        WHERE LOWER(city) = ? AND kosher = 1
        ORDER BY rating DESC
    """
    results = _run_query(query, (city.strip().lower(),))
    if isinstance(results, str):
        return results
    if not results:
        return f"No kosher restaurants found in {city}."
    return json.dumps(results, indent=2)


@tool
def local_transport_guide(city: str) -> str:
    """
    Get local transport options inside a city (metro, bus, taxi, bike-share).
    city: city name (e.g. 'Paris', 'Tokyo', 'New York').
    Returns transport modes with price ranges and practical tips.
    """
    query = """
        SELECT mode, description, price_range, tip
        FROM local_transport
        WHERE LOWER(city) = ?
        ORDER BY mode ASC
    """
    results = _run_query(query, (city.strip().lower(),))
    if isinstance(results, str):
        return results
    if not results:
        return f"No local transport information found for {city}."
    return json.dumps(results, indent=2)


@tool
def airport_transfer_info(city: str) -> str:
    """
    Get airport-to-city transfer options including train, bus, and taxi.
    city: city name (e.g. 'Paris', 'London', 'Tokyo').
    Returns transfer modes with duration, price, and description.
    """
    query = """
        SELECT mode, duration, price_usd, description
        FROM airport_transfers
        WHERE LOWER(city) = ?
        ORDER BY price_usd ASC
    """
    results = _run_query(query, (city.strip().lower(),))
    if isinstance(results, str):
        return results
    if not results:
        return f"No airport transfer information found for {city}."
    return json.dumps(results, indent=2)


@tool
def events_finder(city: str, month: str) -> str:
    """
    Find events, festivals, and things happening in a city during a specific month.
    city: city name (e.g. 'Paris', 'Berlin').
    month: month name in English (e.g. 'june', 'december').
    Returns events with name, category, and description.
    """
    query = """
        SELECT name, category, description
        FROM events
        WHERE LOWER(city) = ? AND LOWER(month) = ?
        ORDER BY name ASC
    """
    results = _run_query(query, (city.strip().lower(), month.strip().lower()))
    if isinstance(results, str):
        return results
    if not results:
        return f"No events found in {city} during {month}."
    return json.dumps(results, indent=2)
