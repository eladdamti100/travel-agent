"""
Context enricher.

This module is responsible for building and improving TripContext.

It has two extraction layers:

Layer 1 — deterministic extraction:
    Fast regex/state-based extraction.
    Runs immediately and never waits for an LLM.

Layer 2 — async SLM enrichment:
    Runs in parallel with other planner work.
    Tries to fill missing context fields and detect stable user preferences.

The master planner should:
    1. call extract_trip_context_deterministic(state)
    2. start enrich_trip_context_async(state, deterministic_context) in parallel
    3. run ready tasks immediately
    4. merge enrichment result when it returns
"""

import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_model
from src.graph.state import AgentState
from src.models.context_enrichment import ContextEnrichmentResult
from src.models.trip_context import (
    DESTINATION_COUNTRY_BY_CITY,
    PERSISTENT_PREFERENCE_FIELDS,
    TripContext,
)
from src.prompts.loader import get_prompt
from src.utils.logger import get_logger

logger = get_logger("context_enricher")


_SUPPORTED_CITY_KEYWORDS = {
    "paris": "Paris",
    "london": "London",
    "tokyo": "Tokyo",
    "new york": "New York",
    "berlin": "Berlin",
}


_COUNTRY_ALIASES = {
    "israel": "Israel",
    "israeli": "Israel",
    "usa": "United States",
    "u.s.": "United States",
    "us": "United States",
    "united states": "United States",
    "america": "United States",
    "american": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "united kingdom": "United Kingdom",
    "england": "United Kingdom",
    "britain": "United Kingdom",
    "british": "United Kingdom",
    "france": "France",
    "french": "France",
    "germany": "Germany",
    "german": "Germany",
    "japan": "Japan",
    "japanese": "Japan",
}




def extract_trip_context_deterministic(state: AgentState) -> TripContext:
    """
    Builds TripContext using only fast deterministic logic.

    Sources:
      - latest user message
      - lightweight metadata already stored in AgentState
      - saved preferences from previous turns

    This function does not call an LLM.
    """
    latest_message = _get_latest_user_message(state)
    text = latest_message.lower()

    destination_city = _extract_destination_city(text) or state.get("current_city")
    destination_country = (
        DESTINATION_COUNTRY_BY_CITY.get(destination_city)
        if destination_city
        else None
    )

    context = TripContext(
        origin_airport=_extract_origin_airport(latest_message),
        origin_country=_extract_origin_country(text),
        destination_city=destination_city,
        destination_country=destination_country,
        duration_days=_extract_duration_days(text),
        total_budget=_extract_total_budget(text) or state.get("total_budget"),
        num_travelers=_extract_num_travelers(text) or state.get("num_travelers"),
        preferred_airline=state.get("preferred_airline"),
        food_preference=state.get("food_preference"),
        travel_preferences=state.get("travel_preferences"),
        hotel_preference=_extract_hotel_preference(text),
        flight_preference=_extract_flight_preference(text),
        activity_preference=_extract_activity_preference(text),
        travel_style=_extract_travel_style(text),
        extraction_source="deterministic",
        slm_enriched=False,
    )

    logger.info("Deterministic TripContext extracted: %s", context.model_dump())
    return context


async def enrich_trip_context_async(
    state: AgentState,
    current_context: TripContext,
) -> ContextEnrichmentResult:
    """
    Asynchronously enriches a deterministic TripContext using a structured SLM call.

    This function should be launched in parallel by the master planner so that
    ready tool tasks can start while enrichment is still running.
    """
    latest_message = _get_latest_user_message(state)

    try:
        model = get_model(temperature=0).with_structured_output(ContextEnrichmentResult)

        response = await model.ainvoke([
            SystemMessage(content=get_prompt("context_enricher_prompt")),
            HumanMessage(
                content=(
                    "Latest user message:\n"
                    f"{latest_message}\n\n"
                    "Current deterministic TripContext:\n"
                    f"{current_context.model_dump()}\n\n"
                    "Saved user preferences from state:\n"
                    f"{_build_saved_preferences_snapshot(state)}"
                )
            ),
        ])

        result = _sanitize_enrichment_result(response)

        logger.info(
            "Context enrichment completed. confidence=%.2f updates=%d",
            result.confidence,
            len(result.preference_updates),
        )

        return result

    except Exception as error:
        logger.error("Context enrichment failed: %s", error)

        fallback_context = current_context.model_copy(
            update={
                "extraction_source": current_context.extraction_source or "deterministic",
                "slm_enriched": False,
            }
        )

        return ContextEnrichmentResult(
            trip_context=fallback_context,
            preference_updates=[],
            confidence=0.0,
            notes=f"Context enrichment failed: {error}",
        )


def merge_trip_context(
    base_context: TripContext,
    enrichment_result: ContextEnrichmentResult,
) -> TripContext:
    """
    Merges SLM-enriched context into the deterministic base context.

    Merge rule:
      - Keep deterministic values when they already exist.
      - Use SLM values only for fields that are missing in the base context.
      - Mark the merged context as slm_enriched when enrichment succeeded.
    """
    enriched = enrichment_result.trip_context
    merged_data = base_context.model_dump()

    for field_name, enriched_value in enriched.model_dump().items():
        if field_name in {"extraction_source", "slm_enriched"}:
            continue

        if merged_data.get(field_name) in (None, "", []):
            if enriched_value not in (None, "", []):
                merged_data[field_name] = enriched_value

    merged_data["extraction_source"] = "merged"
    merged_data["slm_enriched"] = enrichment_result.confidence > 0

    return TripContext(**merged_data)


def _sanitize_enrichment_result(
    result: ContextEnrichmentResult,
) -> ContextEnrichmentResult:
    """
    Removes invalid persistent preference updates and normalizes metadata.
    """
    valid_updates = []

    for update in result.preference_updates:
        if update.field_name not in PERSISTENT_PREFERENCE_FIELDS:
            logger.info(
                "Skipping non-persistent preference update: %s",
                update.field_name,
            )
            continue

        if not update.value:
            continue

        valid_updates.append(update)

    trip_context = result.trip_context.model_copy(
        update={
            "extraction_source": result.trip_context.extraction_source or "slm",
            "slm_enriched": True,
        }
    )

    return ContextEnrichmentResult(
        trip_context=trip_context,
        preference_updates=valid_updates,
        confidence=result.confidence,
        notes=result.notes,
    )


def _get_latest_user_message(state: AgentState) -> str:
    """
    Returns the latest user message content from state.
    """
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)

    messages = state.get("messages", [])
    if not messages:
        return ""

    content = getattr(messages[-1], "content", "")
    return content if isinstance(content, str) else str(content)


def _build_saved_preferences_snapshot(state: AgentState) -> dict:
    """
    Builds a small snapshot of persisted user preferences for enrichment context.
    """
    return {
        "preferred_airline": state.get("preferred_airline"),
        "food_preference": state.get("food_preference"),
        "num_travelers": state.get("num_travelers"),
        "travel_preferences": state.get("travel_preferences"),
    }


def _extract_origin_airport(text: str) -> Optional[str]:
    """
    Extracts a departure airport code from the user message.

    Supported examples:
      - from TLV
      - flying from JFK
      - depart from LHR
      - origin airport TLV
    """
    raw_text = text.upper()

    patterns = [
        r"\bFROM\s+([A-Z]{3})\b",
        r"\bFLYING\s+FROM\s+([A-Z]{3})\b",
        r"\bDEPART(?:ING)?\s+FROM\s+([A-Z]{3})\b",
        r"\bORIGIN\s+AIRPORT\s+([A-Z]{3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text)
        if match:
            return match.group(1)

    return None


def _extract_origin_country(text: str) -> Optional[str]:
    """
    Extracts traveler origin/passport country from simple wording.
    """
    patterns = [
        r"\bpassport(?:\s+country)?\s*(?:is\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
        r"\borigin(?:\s+country)?\s*(?:is\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
        r"\b(?:i\s*am|i'?m)\s+from\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\b(?:i\s*am|i'?m)\s+([a-z]+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue

        raw_country = match.group(1).strip().lower()

        for alias, canonical in _COUNTRY_ALIASES.items():
            if alias in raw_country:
                return canonical
                
        if raw_country not in ["looking", "planning", "going", "flying", "traveling", "a"]:
            return raw_country.title()

    return None


def _extract_destination_city(text: str) -> Optional[str]:
    """
    Extracts a supported destination city from the user message.
    """
    for keyword, city in _SUPPORTED_CITY_KEYWORDS.items():
        if keyword in text:
            return city

    return None


def _extract_duration_days(text: str) -> Optional[int]:
    """
    Extracts trip duration in days or nights.
    """
    patterns = [
        r"\b(\d+)\s*-\s*day\b",
        r"\b(\d+)\s+days?\b",
        r"\b(\d+)\s+nights?\b",
        r"\bfor\s+(\d+)\s+days?\b",
        r"\bfor\s+(\d+)\s+nights?\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return None


def _extract_total_budget(text: str) -> Optional[float]:
    """
    Extracts total budget in USD from common formats.
    """
    patterns = [
        r"\$(\d[\d,]*(?:\.\d+)?)",
        r"(\d[\d,]*(?:\.\d+)?)\s*\$",
        r"\b(?:budget|under|up to|max|maximum)(?:\s+is)?\s+\$?(\d[\d,]*(?:\.\d+)?)\b",
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:usd|dollars)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (ValueError, TypeError):
                continue

    return None


def _extract_num_travelers(text: str) -> Optional[int]:
    """
    Extracts number of travelers.
    """
    patterns = [
        r"\b(\d+)\s*(people|person|traveler|travelers|traveller|travellers|passenger|passengers|of us|pax)\b",
        r"\bwe\s+are\s+(\d+)\b",
        r"\bfamily\s+of\s+(\d+)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))

    return None


def _extract_hotel_preference(text: str) -> Optional[str]:
    """
    Extracts simple hotel preferences.
    """
    if "5-star" in text or "five star" in text:
        return "Prefers 5-star hotels"

    if "cheap hotel" in text or "budget hotel" in text:
        return "Prefers budget hotels"

    if "city center" in text or "central hotel" in text:
        return "Prefers central hotels"

    if "family-friendly hotel" in text:
        return "Prefers family-friendly hotels"

    return None


def _extract_flight_preference(text: str) -> Optional[str]:
    """
    Extracts simple flight preferences.
    """
    if "direct flight" in text or "direct flights" in text:
        return "Direct flights only"

    if "morning flight" in text or "morning flights" in text:
        return "Prefers morning flights"

    if "business class" in text:
        return "Prefers business class"

    if "economy" in text:
        return "Prefers economy class"

    return None


def _extract_activity_preference(text: str) -> Optional[str]:
    """
    Extracts simple activity preferences.
    """
    if "museum" in text or "museums" in text:
        return "Prefers museums"

    if "kids" in text or "children" in text or "family activities" in text:
        return "Prefers family-friendly activities"

    if "nature" in text or "parks" in text:
        return "Prefers nature activities"

    if "shopping" in text:
        return "Prefers shopping"

    return None


def _extract_travel_style(text: str) -> Optional[str]:
    """
    Extracts general travel style.
    """
    if "luxury" in text:
        return "luxury"

    if "budget" in text:
        return "budget"

    if "relaxed" in text or "slow pace" in text:
        return "relaxed"

    if "adventure" in text:
        return "adventure"

    if "family" in text:
        return "family"

    return None
