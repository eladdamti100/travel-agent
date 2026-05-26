from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.trip_context import TripContext


class PreferenceUpdate(BaseModel):
    """
    A persistent user preference discovered by the SLM enrichment layer.

    Only stable preferences should be saved here.
    Trip-specific details such as destination, duration, and budget should stay
    inside TripContext and should not be persisted as user preferences.
    """

    field_name: str = Field(
        description=(
            "The preference field to update, for example preferred_airline, "
            "food_preference, hotel_preference, flight_preference, "
            "activity_preference, travel_style, or travel_preferences."
        )
    )

    value: str = Field(
        description="The preference value to save."
    )

    reason: str = Field(
        description="Short explanation of why this should be treated as a persistent preference."
    )


class ContextEnrichmentResult(BaseModel):
    """
    Structured result returned by the async SLM context enrichment layer.

    The enrichment layer can improve the current TripContext and separately
    suggest persistent preference updates.
    """

    trip_context: TripContext = Field(
        description="Improved trip context for the current planning request."
    )

    preference_updates: List[PreferenceUpdate] = Field(
        default_factory=list,
        description="Stable user preferences that should be persisted for future sessions."
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence of the SLM enrichment result."
    )

    notes: Optional[str] = Field(
        default=None,
        description="Optional short notes about ambiguity or assumptions in the enrichment."
    )