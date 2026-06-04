"""
City Registry — single source of truth for all city/airport/country/currency maps.

All other modules import from here. Adding a new supported city requires editing
only this file.
"""

from typing import Dict, FrozenSet

# ── Supported destination cities ──────────────────────────────────────────────

# Canonical city names (title-cased).
SUPPORTED_CITIES: FrozenSet[str] = frozenset({
    "Paris", "London", "Tokyo", "New York", "Berlin",
})

# Keyword → canonical city name (lowercase keys for case-insensitive matching).
CITY_KEYWORDS: Dict[str, str] = {
    "paris":    "Paris",
    "london":   "London",
    "tokyo":    "Tokyo",
    "new york": "New York",
    "berlin":   "Berlin",
}

# ── City → destination country ────────────────────────────────────────────────

COUNTRY_BY_CITY: Dict[str, str] = {
    "Paris":    "France",
    "London":   "United Kingdom",
    "Tokyo":    "Japan",
    "New York": "United States",
    "Berlin":   "Germany",
}

# ── City ↔ primary IATA airport code ─────────────────────────────────────────

# Canonical lowercase city name → primary IATA code.
AIRPORT_BY_CITY: Dict[str, str] = {
    "paris":    "CDG",
    "london":   "LHR",
    "tokyo":    "NRT",
    "new york": "JFK",
    "berlin":   "BER",
    "tel aviv": "TLV",
    "telaviv":  "TLV",
}

# IATA code → human-readable city name (expanded — includes common origin airports).
CITY_BY_AIRPORT: Dict[str, str] = {
    # North America
    "JFK": "New York", "LGA": "New York", "EWR": "New York",
    "LAX": "Los Angeles", "ORD": "Chicago", "ATL": "Atlanta",
    "DFW": "Dallas", "SFO": "San Francisco", "MIA": "Miami",
    # Europe
    "LHR": "London", "LGW": "London",
    "CDG": "Paris",  "ORY": "Paris",
    "TXL": "Berlin", "BER": "Berlin",
    # Middle East
    "TLV": "Tel Aviv",
    # Asia-Pacific
    "NRT": "Tokyo",  "HND": "Tokyo",
}

# ── City → local (destination) currency ──────────────────────────────────────

CURRENCY_BY_CITY: Dict[str, str] = {
    "london":   "GBP",
    "paris":    "EUR",
    "berlin":   "EUR",
    "tokyo":    "JPY",
    "new york": "USD",
}

# ── Origin/passport country → home currency ──────────────────────────────────

CURRENCY_BY_COUNTRY: Dict[str, str] = {
    "israel":         "ILS",
    "united states":  "USD",
    "usa":            "USD",
    "united kingdom": "GBP",
    "uk":             "GBP",
    "france":         "EUR",
    "germany":        "EUR",
    "japan":          "JPY",
    "australia":      "AUD",
    "canada":         "CAD",
    "india":          "INR",
}

# ── Country name aliases (natural language → canonical) ───────────────────────

COUNTRY_ALIASES: Dict[str, str] = {
    "israel":         "Israel",
    "israeli":        "Israel",
    "usa":            "United States",
    "u.s.":           "United States",
    "us":             "United States",
    "united states":  "United States",
    "america":        "United States",
    "american":       "United States",
    "uk":             "United Kingdom",
    "u.k.":           "United Kingdom",
    "united kingdom": "United Kingdom",
    "england":        "United Kingdom",
    "britain":        "United Kingdom",
    "british":        "United Kingdom",
    "france":         "France",
    "french":         "France",
    "germany":        "Germany",
    "german":         "Germany",
    "japan":          "Japan",
    "japanese":       "Japan",
}

# Normalisation map used by db_tools for visa lookups
# (canonical country → short DB key).
DB_COUNTRY_ALIASES: Dict[str, str] = {
    "united states":        "usa",
    "united kingdom":       "uk",
    "united arab emirates": "uae",
}
