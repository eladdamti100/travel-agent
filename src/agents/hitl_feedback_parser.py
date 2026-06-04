"""
HITL feedback parser — extracts trip parameter overrides from free-text user edits.

Extracted from planner.py._run_master_planner_async to keep the orchestration
function readable.  Called only when force_replan=True and hitl_feedback is set.
"""

import re
from typing import Optional

from src.config.city_registry import (
    AIRPORT_BY_CITY as _CITY_TO_AIRPORT,
    CITY_BY_AIRPORT as _AIRPORT_BY_IATA,
    COUNTRY_BY_CITY as _COUNTRY_BY_CITY,
)
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("hitl_feedback_parser")

_ORIGIN_TRIGGERS = (
    r"(?:flight(?:\s+is)?|fly(?:ing)?|depart(?:ing)?|origin)\s+(?:is\s+)?from\s+",
    r"\bfrom\s+",
    r"\borigin\s+(?:is\s+)?(?:city\s+)?(?:is\s+)?",
    r"\bchange\s+(?:the\s+)?(?:flight\s+)?(?:origin|departure)\s+to\s+",
)

_DEST_TRIGGERS = (
    r"(?:change\s+)?(?:the\s+)?destination\s+to\s+",
    r"\bfly\s+to\s+",
    r"\btravel\s+to\s+",
)

_DUR_PATTERNS = (
    r"\b(\d+)\s*-\s*day\b",
    r"\b(\d+)\s+days?\b",
    r"\b(\d+)\s+nights?\b",
    r"\bfor\s+(\d+)\s+days?\b",
    r"\bfor\s+(\d+)\s+nights?\b",
)


def apply_hitl_feedback(
    context: TripContext,
    hitl_feedback: str,
) -> TripContext:
    """
    Parse hitl_feedback for origin, destination, duration and budget overrides
    and return a new TripContext with the detected changes applied.

    Only called when force_replan=True and hitl_feedback is non-empty.
    """
    if not hitl_feedback:
        return context

    fb_lower = hitl_feedback.lower()
    iata_to_city = {v: k.title() for k, v in _CITY_TO_AIRPORT.items()}

    context = _apply_origin_override(context, hitl_feedback, fb_lower, iata_to_city)
    context = _apply_destination_override(context, fb_lower)
    context = _apply_duration_override(context, fb_lower)
    context = _apply_budget_override(context, fb_lower)
    return context


# ── Private helpers ───────────────────────────────────────────────────────────

def _apply_origin_override(
    context: TripContext,
    hitl_feedback: str,
    fb_lower: str,
    iata_to_city: dict,
) -> TripContext:
    fb_origin_airport: Optional[str] = None
    fb_origin_city: Optional[str] = None

    # Try bare IATA code first (e.g. "change to CDG")
    iata_match = re.search(r"\b([A-Z]{3})\b", hitl_feedback.upper())
    if iata_match:
        iata = iata_match.group(1)
        if iata in iata_to_city:
            fb_origin_airport = iata
            fb_origin_city = iata_to_city[iata]

    # Fall back to city name after an origin-intent phrase
    if not fb_origin_airport:
        for trigger in _ORIGIN_TRIGGERS:
            for city, code in _CITY_TO_AIRPORT.items():
                if re.search(trigger + re.escape(city), fb_lower):
                    fb_origin_airport = code
                    fb_origin_city = city.title()
                    break
            if fb_origin_airport:
                break

    if fb_origin_airport:
        old_airport = context.origin_airport or "?"
        old_city = _AIRPORT_BY_IATA.get(old_airport, old_airport)
        logger.info(
            "HITL edit: origin override %s (%s) → %s (%s)",
            old_airport, old_city,
            fb_origin_airport, fb_origin_city,
        )
        context = context.model_copy(
            update={"origin_airport": fb_origin_airport}, validate=True
        )

    return context


def _apply_destination_override(context: TripContext, fb_lower: str) -> TripContext:
    for trigger in _DEST_TRIGGERS:
        for city in _COUNTRY_BY_CITY:
            if re.search(trigger + re.escape(city.lower()), fb_lower):
                dest_country = _COUNTRY_BY_CITY.get(city)
                logger.info("HITL edit: destination override → %s", city)
                context = context.model_copy(
                    update={"destination_city": city, "destination_country": dest_country},
                    validate=True,
                )
                return context
    return context


def _apply_duration_override(context: TripContext, fb_lower: str) -> TripContext:
    for pat in _DUR_PATTERNS:
        m = re.search(pat, fb_lower)
        if m:
            days = int(m.group(1))
            logger.info("HITL edit: duration override → %d days", days)
            return context.model_copy(update={"duration_days": days}, validate=True)
    return context


def _apply_budget_override(context: TripContext, fb_lower: str) -> TripContext:
    budget_match = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", fb_lower)
    if budget_match:
        budget = float(budget_match.group(1).replace(",", ""))
        logger.info("HITL edit: budget override → $%.2f", budget)
        return context.model_copy(update={"total_budget": budget}, validate=True)
    return context
