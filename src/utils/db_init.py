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

    for table in ("visa_requirements", "activities", "hotels", "flights", "weather",
                  "restaurants", "local_transport", "airport_transfers", "events"):
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

    cursor.execute("""
        CREATE TABLE weather (
            id          INTEGER PRIMARY KEY,
            city        TEXT,
            month       TEXT,
            avg_temp_c  INTEGER,
            description TEXT,
            rainfall    TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE restaurants (
            id          INTEGER PRIMARY KEY,
            city        TEXT,
            name        TEXT,
            cuisine     TEXT,
            price_range TEXT,
            kosher      INTEGER,
            rating      REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE local_transport (
            id            INTEGER PRIMARY KEY,
            city          TEXT,
            mode          TEXT,
            description   TEXT,
            price_range   TEXT,
            tip           TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE airport_transfers (
            id          INTEGER PRIMARY KEY,
            city        TEXT,
            mode        TEXT,
            duration    TEXT,
            price_usd   TEXT,
            description TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE events (
            id          INTEGER PRIMARY KEY,
            city        TEXT,
            month       TEXT,
            name        TEXT,
            category    TEXT,
            description TEXT
        )
    """)

    # ── Flights ───────────────────────────────────────────────────────────────
    flights = [
        # From TLV (Tel Aviv)
        ("TLV", "Paris",    "El Al",           350, "LY321"),
        ("TLV", "Paris",    "Air France",      420, "AF123"),
        ("TLV", "London",   "British Airways", 450, "BA164"),
        ("TLV", "London",   "Virgin Atlantic", 390, "VS100"),
        ("TLV", "Tokyo",    "El Al",           950, "LY091"),
        ("TLV", "Tokyo",    "Emirates",        820, "EK312"),
        ("TLV", "New York", "United",          750, "UA445"),
        ("TLV", "New York", "Delta",           810, "DL402"),
        ("TLV", "Berlin",   "Lufthansa",       280, "LH909"),
        ("TLV", "Berlin",   "Ryanair",         110, "FR101"),
        # From JFK (New York)
        ("JFK", "Paris",    "Air France",      480, "AF007"),
        ("JFK", "Paris",    "Delta",           520, "DL402"),
        ("JFK", "London",   "British Airways", 420, "BA178"),
        ("JFK", "London",   "American",        390, "AA100"),
        ("JFK", "Tokyo",    "Japan Airlines",  850, "JL005"),
        ("JFK", "Tokyo",    "United",          920, "UA837"),
        ("JFK", "Berlin",   "Lufthansa",       510, "LH400"),
        ("JFK", "Berlin",   "Norse Atlantic",  380, "N0701"),
        # From LHR (London Heathrow)
        ("LHR", "Paris",    "Air France",      120, "AF1680"),
        ("LHR", "Paris",    "British Airways", 140, "BA308"),
        ("LHR", "Tokyo",    "British Airways", 760, "BA005"),
        ("LHR", "Tokyo",    "Japan Airlines",  800, "JL043"),
        ("LHR", "New York", "British Airways", 380, "BA117"),
        ("LHR", "New York", "Virgin Atlantic", 360, "VS003"),
        ("LHR", "Berlin",   "British Airways", 130, "BA902"),
        ("LHR", "Berlin",   "Ryanair",          80, "FR9002"),
        # From CDG (Paris Charles de Gaulle)
        ("CDG", "London",   "Air France",      130, "AF1180"),
        ("CDG", "London",   "EasyJet",          90, "U29048"),
        ("CDG", "Tokyo",    "Air France",      750, "AF292"),
        ("CDG", "New York", "Air France",      420, "AF011"),
        ("CDG", "New York", "Delta",           460, "DL264"),
        ("CDG", "Berlin",   "Air France",      110, "AF1220"),
        ("CDG", "Berlin",   "EasyJet",          85, "U22941"),
        # From BER (Berlin Brandenburg)
        ("BER", "Paris",    "Lufthansa",       100, "LH1008"),
        ("BER", "Paris",    "EasyJet",          75, "U22940"),
        ("BER", "London",   "British Airways", 120, "BA903"),
        ("BER", "London",   "Ryanair",          70, "FR9001"),
        ("BER", "Tokyo",    "Lufthansa",       720, "LH796"),
        ("BER", "New York", "Lufthansa",       480, "LH401"),
        ("BER", "New York", "Norse Atlantic",  360, "N0702"),
        # From NRT (Tokyo Narita)
        ("NRT", "Paris",    "Air France",      800, "AF291"),
        ("NRT", "Paris",    "Japan Airlines",  820, "JL415"),
        ("NRT", "London",   "British Airways", 780, "BA006"),
        ("NRT", "New York", "Japan Airlines",  860, "JL004"),
        ("NRT", "New York", "United",          900, "UA838"),
        ("NRT", "Berlin",   "Lufthansa",       730, "LH797"),
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
        # Israel passports
        ("israel", "france",   "No visa required for tourism up to 90 days (Schengen)."),
        ("israel", "japan",    "No visa required for tourism up to 90 days."),
        ("israel", "uk",       "No visa required for tourism up to 6 months."),
        ("israel", "usa",      "ESTA authorization required — apply online before travel."),
        ("israel", "germany",  "No visa required for tourism up to 90 days (Schengen)."),
        # USA passports
        ("usa",    "france",   "No visa required for tourism up to 90 days (Schengen)."),
        ("usa",    "japan",    "No visa required for tourism up to 90 days."),
        ("usa",    "uk",       "No visa required for tourism up to 6 months."),
        ("usa",    "germany",  "No visa required for tourism up to 90 days (Schengen)."),
        # UK passports
        ("uk",     "france",   "No visa required for tourism up to 90 days (Schengen post-Brexit)."),
        ("uk",     "japan",    "No visa required for tourism up to 90 days."),
        ("uk",     "usa",      "ESTA authorization required — apply online before travel."),
        ("uk",     "germany",  "No visa required for tourism up to 90 days (Schengen post-Brexit)."),
        # Indian passports
        ("india",  "france",   "Schengen Visa required. Apply at the French consulate."),
        ("india",  "japan",    "Visa required. Apply at the Japanese embassy."),
        ("india",  "uk",       "Standard Visitor Visa required. Apply online via UK Visas and Immigration."),
        ("india",  "usa",      "B-1/B-2 tourist visa required. Apply at the US Embassy."),
        ("india",  "germany",  "Schengen Visa required. Apply at the German consulate."),
        # French passports
        ("france", "japan",    "No visa required for tourism up to 90 days."),
        ("france", "uk",       "No visa required for tourism up to 6 months."),
        ("france", "usa",      "ESTA authorization required — apply online before travel."),
        # German passports
        ("germany","japan",    "No visa required for tourism up to 90 days."),
        ("germany","uk",       "No visa required for tourism up to 6 months."),
        ("germany","usa",      "ESTA authorization required — apply online before travel."),
        # Japanese passports
        ("japan",  "france",   "No visa required for tourism up to 90 days (Schengen)."),
        ("japan",  "uk",       "No visa required for tourism up to 6 months."),
        ("japan",  "usa",      "ESTA authorization required — apply online before travel."),
        ("japan",  "germany",  "No visa required for tourism up to 90 days (Schengen)."),
    ]
    cursor.executemany(
        "INSERT INTO visa_requirements (origin_country, destination_country, requirement) VALUES (?,?,?)",
        visa_requirements,
    )

    # ── Weather (seasonal averages) ───────────────────────────────────────────
    weather = [
        ("paris",    "december", 6,  "Cold and grey, occasional rain",       "high"),
        ("paris",    "march",    10, "Cool and mild, some showers",           "medium"),
        ("paris",    "june",     22, "Warm and sunny, peak tourist season",   "low"),
        ("paris",    "september",19, "Pleasant, sunny spells",                "low"),
        ("london",   "december", 7,  "Cold, foggy, short days",               "high"),
        ("london",   "march",    9,  "Cool, unpredictable, bring layers",     "medium"),
        ("london",   "june",     19, "Mild and pleasant",                     "medium"),
        ("london",   "september",17, "Mild with some rain",                   "medium"),
        ("tokyo",    "december", 9,  "Cold and dry, great for sightseeing",   "low"),
        ("tokyo",    "march",    12, "Cherry blossom season, very popular",   "medium"),
        ("tokyo",    "june",     24, "Rainy season, hot and humid",           "very high"),
        ("tokyo",    "september",26, "Hot and humid, typhoon risk",           "high"),
        ("new york", "december", 3,  "Cold, possible snow",                   "medium"),
        ("new york", "march",    8,  "Cold to mild, unpredictable",           "low"),
        ("new york", "june",     26, "Hot and humid",                         "low"),
        ("new york", "september",22, "Warm and pleasant, best month to visit","low"),
        ("berlin",   "december", 2,  "Very cold, Christmas markets open",     "medium"),
        ("berlin",   "march",    7,  "Cold to mild, early spring",            "low"),
        ("berlin",   "june",     22, "Warm and sunny, festival season",       "low"),
        ("berlin",   "september",18, "Mild and pleasant",                     "low"),
    ]
    cursor.executemany(
        "INSERT INTO weather (city, month, avg_temp_c, description, rainfall) VALUES (?,?,?,?,?)",
        weather,
    )

    # ── Restaurants ───────────────────────────────────────────────────────────
    # kosher: 1 = kosher certified, 0 = not kosher
    restaurants = [
        ("paris",    "Chez Janou",            "French",       "$$",   0, 4.5),
        ("paris",    "Le Marais Kosher Grill", "Israeli/Grill","$$",   1, 4.3),
        ("paris",    "Café de Flore",          "French Café",  "$$$",  0, 4.2),
        ("paris",    "Moïse Kosher",           "Mediterranean","$$",   1, 4.4),
        ("london",   "Dishoom",                "Indian",       "$$",   0, 4.7),
        ("london",   "Bevis Marks Restaurant", "British/Jewish","$$$", 1, 4.5),
        ("london",   "The Wolseley",           "European",     "$$$",  0, 4.4),
        ("london",   "Tasti Pizza",            "Pizza/Italian","$",    1, 4.1),
        ("tokyo",    "Sukiyabashi Jiro",        "Sushi",        "$$$$", 0, 4.9),
        ("tokyo",    "Ichiran Ramen",           "Ramen",        "$",    0, 4.6),
        ("tokyo",    "Shabusen",                "Shabu-shabu",  "$$",   0, 4.3),
        ("new york", "Katz's Delicatessen",     "Jewish Deli",  "$$",   0, 4.6),
        ("new york", "Le Marais NYC",           "French/Kosher","$$$",  1, 4.4),
        ("new york", "Peter Luger Steak House", "Steakhouse",   "$$$$", 0, 4.7),
        ("new york", "Sushi Yasuda",            "Sushi",        "$$$",  0, 4.6),
        ("berlin",   "Nobelhart & Schmutzig",   "German",       "$$$",  0, 4.5),
        ("berlin",   "Curry 36",                "Street Food",  "$",    0, 4.4),
        ("berlin",   "Borchardt",               "European",     "$$$",  0, 4.3),
        ("berlin",   "Mogg Deli",               "Jewish Deli",  "$$",   1, 4.5),
    ]
    cursor.executemany(
        "INSERT INTO restaurants (city, name, cuisine, price_range, kosher, rating) VALUES (?,?,?,?,?,?)",
        restaurants,
    )

    # ── Local Transport ───────────────────────────────────────────────────────
    local_transport = [
        ("paris",    "Metro",        "14 lines, covers entire city",                    "$2–4/ride",   "Buy a carnet of 10 tickets for a discount."),
        ("paris",    "Bus",          "Extensive network, scenic routes",                "$2–4/ride",   "Same ticket as metro."),
        ("paris",    "Taxi/Uber",    "Widely available, metered",                       "$10–30/trip", "Uber is often cheaper than taxis."),
        ("paris",    "Vélib Bike",   "City bike-share, 1400+ stations",                 "$1–5/day",    "Great for short hops between arrondissements."),
        ("london",   "Tube",         "11 lines, fast city-wide coverage",               "$3–7/ride",   "Use an Oyster card or contactless — never buy paper tickets."),
        ("london",   "Bus",          "Iconic double-deckers, cash not accepted",         "$2/ride",     "Oyster/contactless only."),
        ("london",   "Taxi/Uber",    "Black cabs or Uber widely available",             "$15–40/trip", "Black cabs can be hailed on street; Uber is cheaper."),
        ("london",   "Elizabeth Line","Fast rail across central London",                "$4–8/ride",   "Fastest way between Heathrow and center."),
        ("tokyo",    "Metro/Subway", "13 lines, extremely punctual",                    "$1–3/ride",   "Get a Suica or Pasmo IC card — accepted everywhere."),
        ("tokyo",    "JR Trains",    "Covers greater Tokyo area",                       "$1–5/ride",   "IC card works on all JR lines."),
        ("tokyo",    "Taxi",         "Clean and safe, expensive",                       "$15–50/trip", "Hard to hail; use taxi stands or apps."),
        ("tokyo",    "Bus",          "Less useful for tourists, complex routes",        "$2/ride",     "Stick to metro unless going off the beaten path."),
        ("new york", "Subway",       "24/7 service, 472 stations",                      "$2.90/ride",  "Get an OMNY card or use contactless. Avoid rush hour."),
        ("new york", "Bus",          "Covers areas subway misses",                      "$2.90/ride",  "Same fare as subway, free transfers."),
        ("new york", "Taxi/Uber",    "Yellow cabs everywhere in Manhattan",             "$15–50/trip", "Uber/Lyft often faster outside Manhattan."),
        ("berlin",   "U-Bahn",       "9 underground lines",                             "$3–4/ride",   "Buy a day pass (Tageskarte) for unlimited travel."),
        ("berlin",   "S-Bahn",       "Surface rail, connects suburbs",                  "$3–4/ride",   "Same ticket as U-Bahn — zone system applies."),
        ("berlin",   "Tram",         "Mainly in east Berlin",                           "$3–4/ride",   "Useful for Prenzlauer Berg and Mitte."),
        ("berlin",   "Taxi/Uber",    "Taxis metered, Uber available",                   "$10–25/trip", "Taxis mandatory at night if no other option."),
    ]
    cursor.executemany(
        "INSERT INTO local_transport (city, mode, description, price_range, tip) VALUES (?,?,?,?,?)",
        local_transport,
    )

    # ── Airport Transfers ─────────────────────────────────────────────────────
    airport_transfers = [
        ("paris",    "RER B Train",  "35 min",  "$12",    "CDG → city center. Runs every 10–15 min, stops at Gare du Nord."),
        ("paris",    "Roissybus",    "60 min",  "$17",    "CDG → Opéra. Comfortable, good for heavy luggage."),
        ("paris",    "Taxi",         "45 min",  "$55–70", "Fixed rate from CDG. Agree fare before boarding."),
        ("paris",    "Uber",         "45 min",  "$40–60", "Book in app, available at CDG arrival level."),
        ("london",   "Heathrow Express", "15 min","$35",  "Paddington in 15 minutes. Fastest option, pricey."),
        ("london",   "Elizabeth Line","45 min",  "$14",   "Heathrow → central London. Cheaper than Express."),
        ("london",   "Taxi",         "60 min",  "$60–80", "Black cab from Heathrow. Fixed zones apply."),
        ("london",   "Uber",         "50 min",  "$40–60", "Available at Heathrow, follow signs to pickup zone."),
        ("tokyo",    "Narita Express","60 min",  "$30",   "Narita → Shinjuku/Shibuya. Reserved seating."),
        ("tokyo",    "Limousine Bus", "90 min",  "$20",   "Narita → major hotels. Comfortable, accepts luggage."),
        ("tokyo",    "Taxi",         "90 min",  "$200+",  "Very expensive from Narita — avoid unless necessary."),
        ("tokyo",    "Airport Bus",  "50 min",  "$6",    "Haneda → city center. Cheapest option from Haneda."),
        ("new york", "AirTrain + Subway","60 min","$10",  "JFK → Manhattan via Jamaica station. Cheapest option."),
        ("new york", "NYC Express Bus","75 min", "$18",   "JFK → Midtown. No transfers needed."),
        ("new york", "Taxi",         "60 min",  "$70",   "Flat rate $70 from JFK to Manhattan (plus tolls/tip)."),
        ("new york", "Uber",         "60 min",  "$45–70","Available at JFK. Cheaper than taxi in most cases."),
        ("berlin",   "Airport Express (FEX)","30 min","$4","BER → Hauptbahnhof. Runs every 30 minutes."),
        ("berlin",   "S-Bahn S9",   "45 min",  "$4",    "BER → city. Slower but stops at more stations."),
        ("berlin",   "Taxi",         "40 min",  "$35–50","Metered, queue at airport taxi rank."),
        ("berlin",   "Uber",         "40 min",  "$30–45","Available at BER departures level."),
    ]
    cursor.executemany(
        "INSERT INTO airport_transfers (city, mode, duration, price_usd, description) VALUES (?,?,?,?,?)",
        airport_transfers,
    )

    # ── Events (seasonal) ─────────────────────────────────────────────────────
    events = [
        ("paris",    "june",     "Fête de la Musique",       "Festival",  "Free street music festival across the entire city on June 21."),
        ("paris",    "june",     "Paris Pride",              "Parade",    "LGBTQ+ Pride parade, late June, Marais district."),
        ("paris",    "december", "Christmas Markets",        "Market",    "Traditional markets at Champs-Élysées and La Défense."),
        ("paris",    "march",    "Paris Fashion Week",       "Fashion",   "Ready-to-wear collections, late February to early March."),
        ("paris",    "september","Journées du Patrimoine",   "Culture",   "Heritage Days — free entry to monuments and palaces."),
        ("london",   "june",     "Trooping the Colour",      "Royal",     "The King's Birthday Parade, Horse Guards Parade."),
        ("london",   "june",     "Wimbledon",                "Sports",    "Grand Slam tennis tournament, late June to early July."),
        ("london",   "december", "Winter Wonderland",        "Festival",  "Hyde Park Christmas fair, ice rink, and rides."),
        ("london",   "march",    "St Patrick's Day Parade",  "Parade",    "Large Irish community parade through central London."),
        ("tokyo",    "march",    "Cherry Blossom (Hanami)",  "Nature",    "Peak bloom late March–early April. Parks fill with picnickers."),
        ("tokyo",    "june",     "Sanno Matsuri Festival",   "Festival",  "One of Tokyo's three great festivals, Chiyoda area."),
        ("tokyo",    "december", "Illuminations",            "Lights",    "City-wide Christmas light displays throughout December."),
        ("tokyo",    "september","Tokyo Game Show",          "Tech",      "Massive gaming expo at Makuhari Messe."),
        ("new york", "june",     "NYC Pride March",          "Parade",    "One of the world's largest Pride events, Fifth Avenue."),
        ("new york", "september","US Open Tennis",           "Sports",    "Grand Slam tournament at Flushing Meadows, late August–September."),
        ("new york", "december", "New Year's Eve Times Square","Festival","Ball drop, crowds of 50,000+. Book accommodation months ahead."),
        ("new york", "march",    "St Patrick's Day Parade",  "Parade",    "Largest in the world, Fifth Avenue, March 17."),
        ("berlin",   "june",     "Berlin Music Week",        "Festival",  "Electronic and indie music events across the city."),
        ("berlin",   "december", "Christmas Markets",        "Market",    "Over 80 markets citywide. Gendarmenmarkt is the most famous."),
        ("berlin",   "march",    "Berlin Fashion Week",      "Fashion",   "International designers and local brands, early January/July."),
        ("berlin",   "september","Berlin Marathon",          "Sports",    "World Major marathon through city center, late September."),
    ]
    cursor.executemany(
        "INSERT INTO events (city, month, name, category, description) VALUES (?,?,?,?,?)",
        events,
    )

    conn.commit()
    conn.close()
    print(f"Database ready: {DB_PATH}")


if __name__ == "__main__":
    create_travel_db()