"""
Canned API responses for mocking external HTTP calls in tests.

Each dict mirrors the real API's response shape exactly so tests exercise
the same JSON-parsing code paths as production.
"""

# ── OpenCage geocoding ────────────────────────────────────────────────────────
OPENCAGE_PARIS = {
    "results": [{"geometry": {"lat": 48.8566, "lng": 2.3522}}],
    "status": {"code": 200},
}
OPENCAGE_EMPTY = {"results": [], "status": {"code": 200}}

# ── Ticketmaster events ───────────────────────────────────────────────────────
TICKETMASTER_EMPTY: dict = {}  # no _embedded key → triggers empty-events fallback

TICKETMASTER_PARIS = {
    "_embedded": {
        "events": [
            {
                "name": "Paris Jazz Festival",
                "dates": {"start": {"localDate": "2099-07-15"}},
                "priceRanges": [{"min": 25.0, "currency": "EUR"}],
            },
            {
                "name": "Louvre Night Tour",
                "dates": {"start": {"localDate": "2099-08-01"}},
                "priceRanges": [{"min": 40.0, "currency": "EUR"}],
            },
        ]
    }
}
TICKETMASTER_EMPTY = {}

# ── ExchangeRate-API ──────────────────────────────────────────────────────────
EXCHANGE_USD_EUR = {
    "result": "success",
    "conversion_rates": {"EUR": 0.92, "GBP": 0.78, "JPY": 156.0, "ILS": 3.75, "USD": 1.0},
}

# ── Open Brewery DB ───────────────────────────────────────────────────────────
BREWERIES_PARIS = [
    {"name": "Brasserie de la Tour", "brewery_type": "micro", "address_1": "10 Rue de Rivoli"},
    {"name": "Le Petit Houblon",      "brewery_type": "taproom", "address_1": "22 Rue Oberkampf"},
]
BREWERIES_EMPTY: list = []

# ── RestCountries ─────────────────────────────────────────────────────────────
RESTCOUNTRIES_FRANCE = [
    {
        "name": {"common": "France"},
        "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
        "region": "Europe",
    }
]
RESTCOUNTRIES_NOT_FOUND: list = []

# ── Tavily search ─────────────────────────────────────────────────────────────
TAVILY_PARIS = [
    {
        "url": "https://example.com/paris-tips",
        "content": "Paris is known for its museums and cafés. Best visited in spring.",
    },
    {
        "url": "https://example.com/paris-weather",
        "content": "Average temperature in Paris in June is 22°C.",
    },
]
TAVILY_EMPTY: list = []
