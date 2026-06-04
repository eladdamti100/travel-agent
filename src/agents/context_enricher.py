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
from src.config.city_registry import (
    AIRPORT_BY_CITY as _CITY_TO_AIRPORT,
    CITY_BY_AIRPORT as _AIRPORT_TO_CITY,
    CITY_KEYWORDS as _SUPPORTED_CITY_KEYWORDS,
    COUNTRY_ALIASES as _COUNTRY_ALIASES,
)
from src.graph.state import AgentState
from src.models.context_enrichment import ContextEnrichmentResult
from src.models.trip_context import (
    DESTINATION_COUNTRY_BY_CITY,
    PERSISTENT_PREFERENCE_FIELDS,
    TripContext,
)
from src.prompts.loader import get_prompt
from src.utils.logger import get_logger
from src.utils.token_tracker import log_token_usage

logger = get_logger("context_enricher")


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
        currency=_extract_currency(latest_message) or state.get("currency"),
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
        log_token_usage(response, call_site="context_enricher")

        result = _sanitize_enrichment_result(response)

        logger.info(
            "Context enrichment completed. confidence=%.2f updates=%d",
            result.confidence,
            len(result.preference_updates),
        )

        return result

    except Exception as error:
        logger.warning("Context enrichment failed (non-critical, using deterministic context): %s", error)

        fallback_context = current_context.model_copy(
            update={
                "extraction_source": current_context.extraction_source or "deterministic",
                "slm_enriched": False,
            },
            validate=True,
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

def merge_modified_trip_context(
    *,
    old_context: TripContext,
    modified_context: TripContext,
) -> TripContext:
    """
    Merges a modified TripContext into an existing trip context.

    Only non-empty modified fields overwrite existing values.
    """
    merged = old_context.model_dump()

    for field_name, value in modified_context.model_dump().items():
        if value in (None, "", []):
            continue

        merged[field_name] = value

    merged["extraction_source"] = "replanned"

    logger.info(
        "Merged modified TripContext. old=%s modified=%s merged=%s",
        old_context.model_dump(),
        modified_context.model_dump(),
        merged,
    )

    return TripContext(**merged)


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
        },
        validate=True,
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
      - from TLV / form TLV (typo)
      - flying from JFK / fly from LHR / fly form TLV
      - depart from LHR
      - origin airport TLV
      - TLV (bare code)
      - TLV, Israeli passport (code-first)
      - airport: TLV / airport is TLV / airport TLV
    """
    raw_text = text.upper()

    # Words that look like a 3-letter code but are really the start of a known
    # city name (e.g. "from New York" → "NEW", "from Tel Aviv" → "TEL").
    _NOT_AIRPORT_PREFIXES = (
        {kw.upper()[:3] for kw in _SUPPORTED_CITY_KEYWORDS}
        | {city.upper()[:3] for city in _CITY_TO_AIRPORT if " " in city}
    )

    patterns = [
        # "from TLV" or "form TLV" (common typo)
        r"\b(?:FROM|FORM)\s+([A-Z]{3})\b",
        # "flying from/form TLV" or "fly from/form TLV"
        r"\bFLY(?:ING)?\s+(?:FROM|FORM)\s+([A-Z]{3})\b",
        # "departing from TLV"
        r"\bDEPART(?:ING)?\s+(?:FROM|FORM)\s+([A-Z]{3})\b",
        # "origin airport TLV"
        r"\bORIGIN\s+AIRPORT\s+([A-Z]{3})\b",
        # "airport: TLV" / "airport is TLV" / "airport TLV"
        r"\bAIRPORT\s*(?:IS|:)?\s*([A-Z]{3})\b",
        # bare code on its own line (HITL short reply: "TLV")
        r"^([A-Z]{3})$",
        # code-first with punctuation/space (HITL short reply: "TLV.")
        r"^([A-Z]{3})[,\.\s]",
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text)
        if match:
            code = match.group(1)
            # Skip false positives like "NEW" from "New York".
            if code in _NOT_AIRPORT_PREFIXES:
                continue
            return code

    # Fallback: origin specified as city name after "from" (e.g. "from New York").
    # Only match when the city follows an explicit origin marker — never match the
    # destination city ("trip to Paris from JFK" must not return CDG here).
    lower_text = text.lower()
    _origin_prefix = (
        r"\b(?:from|form|fly(?:ing)?\s+(?:from|form)"
        r"|depart(?:ing)?\s+(?:from|form)"
        r"|origin(?:\s+airport)?(?:\s+is)?)\s+"
    )
    for city, code in _CITY_TO_AIRPORT.items():
        if re.search(_origin_prefix + re.escape(city) + r"\b", lower_text):
            return code

    return None


def _extract_origin_country(text: str) -> Optional[str]:
    """
    Extracts traveler origin/passport country from simple wording.
    """
    patterns = [
        r"\bpassport(?:\s+country)?\s*(?:is\s+)?([a-z]+(?:\s+[a-z]+)?)\b",
        r"\b([a-z]+(?:\s+[a-z]+)?)\s+passport\b",
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

    # Fallback: bare country word with no sentence context (e.g. HITL reply "Israel")
    for alias, canonical in _COUNTRY_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", text):
            return canonical

    return None


# _AIRPORT_TO_CITY is imported directly from city_registry (CITY_BY_AIRPORT)


def _extract_destination_city(text: str) -> Optional[str]:
    """
    Extracts a supported destination city from the user message.

    Position-aware: when the message names several supported cities (e.g.
    "to Berlin from New York"), the city introduced by "to" is the destination
    and the city introduced by "from" is the origin. Falls back to the earliest
    mentioned city when no "to"/"from" markers are present.

    Also handles IATA codes used as destination ("to TLV" → "Tel Aviv").
    """
    # Check for "to [IATA]" pattern before city-name scan (text is already lowercase).
    iata_dest = re.search(r"\bto\s+([a-zA-Z]{3})\b", text, re.IGNORECASE)
    if iata_dest:
        code = iata_dest.group(1).upper()
        city = _AIRPORT_TO_CITY.get(code)
        if city:
            return city

    # Find every supported city and where it appears in the text.
    found = [
        (text.find(keyword), city)
        for keyword, city in _SUPPORTED_CITY_KEYWORDS.items()
        if keyword in text
    ]
    if not found:
        return None

    if len(found) == 1:
        return found[0][1]

    # Multiple cities — prefer the one that sits right after a "to " marker
    # and is not the one sitting after a "from "/"form " marker.
    to_pos = _last_marker_pos(text, ("to ",))
    from_pos = _last_marker_pos(text, ("from ", "form "))

    if to_pos is not None:
        # Destination is the first supported city appearing after "to ".
        after_to = sorted(
            (pos, city) for pos, city in found if pos >= to_pos
        )
        if after_to:
            dest_pos, dest_city = after_to[0]
            # Guard: if that same city is what "from" points to, skip it.
            if from_pos is None or dest_pos < from_pos or dest_pos != _city_pos_after(text, from_pos, found):
                return dest_city

    # No usable "to" marker — return the earliest mentioned city, but drop the
    # one that clearly belongs to "from".
    origin_city = _city_pos_after(text, from_pos, found) if from_pos is not None else None
    for pos, city in sorted(found):
        if city != origin_city:
            return city

    return found[0][1]


def _last_marker_pos(text: str, markers: tuple) -> Optional[int]:
    """Returns the position just after the last occurrence of any marker, or None."""
    best = None
    for marker in markers:
        idx = text.rfind(marker)
        if idx != -1:
            end = idx + len(marker)
            if best is None or end > best:
                best = end
    return best


def _city_pos_after(text: str, marker_pos: Optional[int], found: list) -> Optional[str]:
    """Returns the first supported city appearing at/after marker_pos."""
    if marker_pos is None:
        return None
    candidates = sorted((pos, city) for pos, city in found if pos >= marker_pos)
    return candidates[0][1] if candidates else None


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
            value = int(match.group(1))
            if value <= 0:
                return None
            return value

    return None


def _extract_total_budget(text: str) -> Optional[float]:
    """
    Extracts total budget amount from common formats (currency-agnostic).
    """
    patterns = [
        r"\$(\d[\d,]*(?:\.\d+)?)",
        r"(\d[\d,]*(?:\.\d+)?)\s*\$",
        r"[€£₪¥](\d[\d,]*(?:\.\d+)?)",
        r"(\d[\d,]*(?:\.\d+)?)\s*[€£₪¥]",
        r"\b(?:budget|under|up to|max|maximum)(?:\s+is)?\s+[\$€£₪¥]?(\d[\d,]*(?:\.\d+)?)\b",
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:usd|eur|gbp|ils|jpy|aud|cad|dollars?|euros?|pounds?|shekels?|yen)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except (ValueError, TypeError):
                continue

    return None


def _extract_currency(text: str) -> Optional[str]:
    """
    Extracts ISO-4217 currency code from budget-related phrases.
    Returns None when USD (the default) so callers only get a value for non-USD.
    """
    t = text.lower()

    # Symbol and keyword detection — ordered most-specific first.
    if "€" in text or " eur" in t or "euro" in t:
        return "EUR"
    if "£" in text or " gbp" in t or "pound" in t:
        return "GBP"
    if "₪" in text or " ils" in t or "shekel" in t or " nis" in t:
        return "ILS"
    if "¥" in text or " jpy" in t or " yen" in t:
        return "JPY"
    if " aud" in t or "australian dollar" in t:
        return "AUD"
    if " cad" in t or "canadian dollar" in t:
        return "CAD"

    # Explicit 3-letter code after a number, e.g. "2000 GBP"
    match = re.search(r"\b\d[\d,]*\s+([A-Z]{3})\b", text)
    if match:
        return match.group(1).upper()

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

    # Exclude "budget $N" / "budget €N" patterns — that's a financial amount, not travel style.
    if re.search(r"\bbudget\b(?!\s*[\$€£₪¥\d])", text):
        return "budget"

    if "relaxed" in text or "slow pace" in text:
        return "relaxed"

    if "adventure" in text:
        return "adventure"

    if "family" in text:
        return "family"

    return None
