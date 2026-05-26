"""
One-time script to create and populate travel_agency.db in data/.
Run once before starting the agent:  python -m src.utils.db_init
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "travel_agency.db"


def create_travel_db() -> None:
    """
    Recreates the local SQLite travel database with seed data.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for table in ("visa_requirements", "activities", "hotels", "flights"):
        cursor.execute(f"DROP TABLE IF EXISTS {table}")

    cursor.execute("""
        CREATE TABLE flights (
            id             INTEGER PRIMARY KEY,
            origin         TEXT,
            destination    TEXT,
            airline        TEXT,
            price          INTEGER,
            flight_number  TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE hotels (
            id               INTEGER PRIMARY KEY,
            city             TEXT,
            name             TEXT,
            price_per_night  INTEGER,
            stars            INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE activities (
            id        INTEGER PRIMARY KEY,
            city      TEXT,
            name      TEXT,
            category  TEXT,
            price     INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE visa_requirements (
            id                  INTEGER PRIMARY KEY,
            origin_country      TEXT,
            destination_country TEXT,
            requirement         TEXT
        )
    """)

    # ── Flights (all from TLV) ────────────────────────────────────────────────
    flights = [
        ("TLV", "Paris",    "El Al",          350, "LY321"),
        ("TLV", "Paris",    "Air France",     420, "AF123"),
        ("TLV", "London",   "British Airways",450, "BA164"),
        ("TLV", "London",   "Virgin Atlantic",390, "VS100"),
        ("TLV", "Tokyo",    "El Al",          950, "LY091"),
        ("TLV", "Tokyo",    "Emirates",       820, "EK312"),
        ("TLV", "New York", "United",         750, "UA445"),
        ("TLV", "Berlin",   "Lufthansa",      280, "LH909"),
        ("TLV", "Berlin",   "Ryanair",        110, "FR101"),
    ]
    cursor.executemany(
        "INSERT INTO flights (origin, destination, airline, price, flight_number) VALUES (?,?,?,?,?)",
        flights,
    )

    # ── Hotels ────────────────────────────────────────────────────────────────
    hotels = [
        ("paris",    "Hotel de Ville",        150, 3),
        ("paris",    "Luxury Ritz",           600, 5),
        ("paris",    "Ibis Budget Paris",      85, 2),
        ("london",   "The Savoy",             450, 5),
        ("london",   "Premier Inn London",    120, 3),
        ("tokyo",    "Shibuya Capsule",        50, 2),
        ("tokyo",    "Park Hyatt Tokyo",      700, 5),
        ("new york", "The Plaza",             850, 5),
        ("new york", "Broadway Hotel",        190, 3),
        ("berlin",   "Berlin Central Hostel",  40, 1),
        ("berlin",   "Hilton Berlin",         220, 4),
    ]
    cursor.executemany(
        "INSERT INTO hotels (city, name, price_per_night, stars) VALUES (?,?,?,?)",
        hotels,
    )

    # ── Activities ────────────────────────────────────────────────────────────
    activities = [
        ("paris",    "Louvre Museum",       "Culture",       20),
        ("paris",    "Eiffel Tower",        "Sightseeing",   35),
        ("paris",    "Disneyland Paris",    "Family",        95),
        ("london",   "London Eye",          "Sightseeing",   30),
        ("london",   "British Museum",      "Culture",        0),
        ("tokyo",    "Robot Cafe",          "Entertainment", 60),
        ("tokyo",    "Mount Fuji Day Trip", "Nature",       120),
        ("new york", "Statue of Liberty",   "Sightseeing",   25),
        ("berlin",   "Berlin Wall Tour",    "History",       15),
        ("berlin",   "Techno Club Entry",   "Nightlife",     25),
    ]
    cursor.executemany(
        "INSERT INTO activities (city, name, category, price) VALUES (?,?,?,?)",
        activities,
    )

    # ── Visa Requirements ─────────────────────────────────────────────────────
    visa_requirements = [
        ("israel", "france",   "No visa required for tourism up to 90 days (Schengen)."),
        ("israel", "japan",    "No visa required for tourism up to 90 days."),
        ("israel", "uk",       "No visa required for tourism up to 6 months."),
        ("israel", "usa",      "ESTA authorization required — apply online before travel."),
        ("israel", "germany",  "No visa required for tourism up to 90 days (Schengen)."),
        ("india",  "france",   "Schengen Visa required. Apply at the French consulate."),
        ("india",  "japan",    "Visa required. Apply at the Japanese embassy."),
        ("usa",    "france",   "No visa required for tourism up to 90 days (Schengen)."),
        ("usa",    "japan",    "No visa required for tourism up to 90 days."),
        ("usa",    "uk",       "No visa required for tourism up to 6 months."),
    ]
    cursor.executemany(
        "INSERT INTO visa_requirements (origin_country, destination_country, requirement) VALUES (?,?,?)",
        visa_requirements,
    )

    conn.commit()
    conn.close()
    print(f"Database ready: {DB_PATH}")


if __name__ == "__main__":
    create_travel_db()
