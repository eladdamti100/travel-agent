"""
FinalPlan model — structured representation of a completed travel plan.

Produced by plan_generator.py after all planner sub-agents have run.
Consumed by the FastAPI layer, the critic, and the PDF exporter.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TripSummary:
    origin_airport: str = ""
    origin_city: Optional[str] = None
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    duration_days: Optional[int] = None
    travel_month: Optional[str] = None
    total_budget: Optional[float] = None
    currency: str = "USD"
    travel_style: Optional[str] = None


@dataclass
class FlightOption:
    airline: str = ""
    flight_number: str = ""
    price: Optional[float] = None


@dataclass
class HotelOption:
    name: str = ""
    price_per_night: Optional[float] = None
    stars: Optional[int] = None


@dataclass
class ActivityItem:
    name: str = ""
    category: str = ""
    price: Optional[float] = None


@dataclass
class CostSummary:
    currency: str = "USD"
    flight_cost: Optional[float] = None
    hotel_total: Optional[float] = None
    activities_total: Optional[float] = None
    estimated_total: Optional[float] = None
    within_budget: Optional[bool] = None


@dataclass
class WebEnrichment:
    coordinates: Optional[Dict[str, float]] = None
    currency_rate: Optional[str] = None
    live_events: List[str] = field(default_factory=list)
    web_highlights: List[str] = field(default_factory=list)


@dataclass
class FinalPlan:
    trip_summary: TripSummary = field(default_factory=TripSummary)
    flights: List[FlightOption] = field(default_factory=list)
    hotels: List[HotelOption] = field(default_factory=list)
    activities: List[ActivityItem] = field(default_factory=list)
    visa_info: str = ""
    cost_summary: CostSummary = field(default_factory=CostSummary)
    web_enrichment: WebEnrichment = field(default_factory=WebEnrichment)
    notes: str = ""
    planning_mode: str = "full_planning"
    used_web_source: bool = False
    raw_markdown: str = ""

    def model_dump(self) -> dict:
        """Pydantic-compatible serialisation for callers that expect model_dump()."""
        return dataclasses.asdict(self)
