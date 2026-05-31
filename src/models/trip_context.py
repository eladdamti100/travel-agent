"""
Trip context model — structured representation of a user trip request.
"""

from typing import Optional

from pydantic import BaseModel, Field


class TripContext(BaseModel):
    """
    Structured trip context extracted from the user request and saved memory.

    The master planner uses this model to decide:
      - which required fields are missing
      - which tools can run immediately
      - which tasks must wait for more user information

    Extraction strategy:
      - deterministic layer: fast regex/state-based extraction
      - SLM enrichment layer: async structured enrichment that may improve context later
    """

    origin_airport: Optional[str] = Field(
        default=None,
        description="Departure airport code, for example TLV, JFK, LHR.",
    )

    origin_country: Optional[str] = Field(
        default=None,
        description="Traveler passport/origin country, used for visa requirements.",
    )

    destination_city: Optional[str] = Field(
        default=None,
        description="Destination city, for example Paris, London, Tokyo, New York, Berlin.",
    )

    destination_country: Optional[str] = Field(
        default=None,
        description="Destination country derived from the destination city when possible.",
    )

    duration_days: Optional[int] = Field(
        default=None,
        description="Trip duration in days or nights, depending on user wording.",
    )

    total_budget: Optional[float] = Field(
        default=None,
        description="Total trip budget in the user's stated currency.",
    )

    currency: Optional[str] = Field(
        default=None,
        description="ISO-4217 budget currency code, e.g. USD, EUR, GBP. Defaults to USD when unspecified.",
    )

    travel_month: Optional[str] = Field(
        default=None,
        description="Month of travel in lowercase English, for example june, december.",
    )

    num_travelers: Optional[int] = Field(
        default=None,
        description="Number of travelers.",
    )

    preferred_airline: Optional[str] = Field(
        default=None,
        description="User's preferred airline, if known.",
    )

    food_preference: Optional[str] = Field(
        default=None,
        description="User's dietary preference, for example kosher, vegan, halal.",
    )

    travel_preferences: Optional[str] = Field(
        default=None,
        description="Additional saved free-form travel preferences.",
    )

    hotel_preference: Optional[str] = Field(
        default=None,
        description="Hotel preference such as 5-star, cheap, central location, family-friendly.",
    )

    flight_preference: Optional[str] = Field(
        default=None,
        description="Flight preference such as direct flights, morning flights, business class.",
    )

    activity_preference: Optional[str] = Field(
        default=None,
        description="Activity preference such as museums, kids activities, nature, shopping.",
    )

    travel_style: Optional[str] = Field(
        default=None,
        description="General travel style such as budget, luxury, relaxed, family, adventure.",
    )

    extraction_source: str = Field(
        default="deterministic",
        description="Source of the current context: deterministic, slm, or merged.",
    )

    slm_enriched: bool = Field(
        default=False,
        description="Whether this context was enriched by the async SLM enrichment layer.",
    )


DESTINATION_COUNTRY_BY_CITY = {
    "Paris": "France",
    "London": "United Kingdom",
    "Tokyo": "Japan",
    "New York": "United States",
    "Berlin": "Germany",
}


REQUIRED_TRIP_FIELDS = (
    "origin_airport",
    "origin_country",
    "destination_city",
    "duration_days",
    "total_budget",
)


PERSISTENT_PREFERENCE_FIELDS = (
    "preferred_airline",
    "food_preference",
    "travel_preferences",
    "hotel_preference",
    "flight_preference",
    "activity_preference",
    "travel_style",
)