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
        (origin_country.strip().lower(), destination_country.strip().lower()),
    )
    if isinstance(results, str):
        return results
    if not results:
        return (
            f"No visa information found for {origin_country} → {destination_country}. "
            "Please check the official embassy website."
        )
    return results[0]["requirement"]
