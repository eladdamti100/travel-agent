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


def _validated_copy(context: TripContext, **updates) -> TripContext:
    """Return a validated TripContext with the given fields overridden."""
    return TripContext.model_validate({**context.model_dump(), **updates})


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

    fb_lower = hitl_feedback.lower().strip()

    context = _apply_origin_override(context, hitl_feedback, fb_lower)
    context = _apply_destination_override(context, fb_lower)
    context = _apply_duration_override(context, fb_lower)
    context = _apply_budget_override(context, fb_lower)
    return context


# ── Private helpers ───────────────────────────────────────────────────────────

def _apply_origin_override(
    context: TripContext,
    hitl_feedback: str,
    fb_lower: str,
) -> TripContext:
    iata_to_city = {v: k.title() for k, v in _CITY_TO_AIRPORT.items()}

    # 1. Bare IATA code (e.g. "CDG", "LHR")
    iata_match = re.search(r"\b([A-Z]{3})\b", hitl_feedback.upper())
    if iata_match:
        iata = iata_match.group(1)
        if iata in iata_to_city:
            logger.info("HITL edit: origin override via IATA -> %s", iata)
            return _validated_copy(context, origin_airport=iata)

    # 2. City name anywhere after origin-intent words
    origin_keywords = (
        "from", "origin", "depart", "departing", "departure",
        "fly from", "flying from", "flight from", "leaving from", "leaving",
        "change origin", "change departure", "set origin", "set departure",
    )
    for city, code in _CITY_TO_AIRPORT.items():
        city_l = city.lower()
        for kw in origin_keywords:
            # e.g. "from paris", "fly from london", "origin paris"
            if re.search(rf"\b{re.escape(kw)}\b.{{0,20}}\b{re.escape(city_l)}\b", fb_lower):
                logger.info("HITL edit: origin override -> %s (%s)", city, code)
                return _validated_copy(context, origin_airport=code)

    return context


def _apply_destination_override(context: TripContext, fb_lower: str) -> TripContext:
    dest_keywords = (
        "to", "destination", "going to", "travel to", "fly to",
        "visit", "visiting", "change destination", "instead of",
        "trip to", "headed to", "head to",
    )
    for city in _COUNTRY_BY_CITY:
        city_l = city.lower()
        for kw in dest_keywords:
            if re.search(rf"\b{re.escape(kw)}\b.{{0,20}}\b{re.escape(city_l)}\b", fb_lower):
                dest_country = _COUNTRY_BY_CITY.get(city)
                logger.info("HITL edit: destination override -> %s", city)
                return _validated_copy(
                    context,
                    destination_city=city,
                    destination_country=dest_country,
                )
    return context


def _apply_duration_override(context: TripContext, fb_lower: str) -> TripContext:
    # Matches: "7 days", "7 nights", "7-day", "for 7 days", "make it 10 days",
    #          "10 day trip", "a week" (=7), "two weeks" (=14)
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "a week": 7, "one week": 7, "two weeks": 14,
    }
    for word, num in word_to_num.items():
        if re.search(rf"\b{re.escape(word)}\b.{{0,10}}\b(?:days?|nights?|day trip|night trip)\b", fb_lower):
            logger.info("HITL edit: duration override (word) -> %d days", num)
            return _validated_copy(context, duration_days=num)

    patterns = (
        r"\b(\d+)\s*[-\s]?\s*(?:day|days|night|nights)\b",
        r"\bfor\s+(\d+)\s+(?:days?|nights?)\b",
        r"\bmake\s+it\s+(\d+)\b",
        r"\bextend\s+(?:to\s+)?(\d+)\b",
    )
    for pat in patterns:
        m = re.search(pat, fb_lower)
        if m:
            days = int(m.group(1))
            if 1 <= days <= 365:
                logger.info("HITL edit: duration override -> %d days", days)
                return _validated_copy(context, duration_days=days)

    return context


def _apply_budget_override(context: TripContext, fb_lower: str) -> TripContext:
    # Matches all natural forms:
    # "$300", "300$", "300 dollars", "300 usd", "300 eur", "300 gbp",
    # "budget of 300", "budget 300", "budget to 300",
    # "increase budget to 500", "set budget 400",
    # "with 280 dollars", "the same trip with budget 280 dollars"
    patterns = (
        r"\$\s*(\d[\d,]*(?:\.\d+)?)",                           # $300
        r"(\d[\d,]*(?:\.\d+)?)\s*\$",                           # 300$
        r"(\d[\d,]*(?:\.\d+)?)\s+(?:dollars?|usd|eur|gbp)",    # 300 dollars
        r"budget\s+(?:of\s+|to\s+|is\s+)?(\d[\d,]*(?:\.\d+)?)",  # budget of 300
        r"(?:increase|raise|set|change|update)\s+(?:the\s+)?budget\s+(?:to\s+)?(\d[\d,]*(?:\.\d+)?)",
        r"with\s+(?:a\s+)?(?:budget\s+(?:of\s+)?)?(\d[\d,]*(?:\.\d+)?)\s*(?:dollars?|usd|eur|gbp)?",
    )
    for pat in patterns:
        m = re.search(pat, fb_lower)
        if m:
            try:
                budget = float(m.group(1).replace(",", ""))
                if budget > 0:
                    logger.info("HITL edit: budget override -> $%.2f", budget)
                    return _validated_copy(context, total_budget=budget)
            except (ValueError, IndexError):
                continue

    return context
