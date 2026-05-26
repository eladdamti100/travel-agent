from typing import Dict, List, Optional
import os
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.graph.state import AgentState
from src.models.preferences import PREFERENCE_EXTRACTION_CONFIG
from src.utils.logger import get_logger

logger = get_logger("preferences_memory_agent")


def run_preferences_memory(state: AgentState) -> dict:
    """
    Handles the preferences and memory route selected by the master orchestrator.

    This route is responsible for default user-memory conversations:
      - Recalling saved preferences from persisted graph state.
      - Updating saved travel preferences from the latest user message.

    This function does not plan trips and does not call travel tools.
    """
    messages = state.get("messages", [])
    if not messages:
        return _run_recall(state)

    last_content = getattr(messages[-1], "content", "")
    last_content_lower = last_content.lower()

    if _is_update_request(last_content_lower):
        return _update_preferences(state, last_content)

    if _is_recall_request(last_content_lower):
        return _run_recall(state)

    logger.info("Preferences memory intent unclear; defaulting to recall.")
    return _run_recall(state)


def _is_update_request(message: str) -> bool:
    """
    Returns True when the user appears to state or update a travel preference.
    """
    return any(
        trigger in message
        for trigger in PREFERENCE_EXTRACTION_CONFIG.update_triggers
    )


def _is_recall_request(message: str) -> bool:
    """
    Returns True when the user asks about remembered preferences or profile data.
    """
    return any(
        trigger in message
        for trigger in PREFERENCE_EXTRACTION_CONFIG.recall_triggers
    )


def _run_recall(state: AgentState) -> dict:
    """
    Answers questions about saved preferences and travel memory directly from state.

    This function does not call an LLM and does not execute tools.
    It only reads persisted graph state and returns a user-facing answer.
    """
    parts = []

    airline = state.get("preferred_airline")
    food = state.get("food_preference")
    travelers = state.get("num_travelers")
    city = state.get("current_city")
    budget = state.get("total_budget")
    extra_preferences = state.get("travel_preferences")

    if airline:
        parts.append(f"- Preferred airline: **{airline}**")

    if food:
        parts.append(f"- Food preference: **{food}**")

    if travelers:
        parts.append(f"- Number of travelers: **{travelers}**")

    if extra_preferences:
        parts.append(f"- Other preferences:\n{extra_preferences}")

    if city:
        parts.append(f"- Last destination: **{city}**")

    if budget:
        parts.append(f"- Last budget: **${budget:,.0f}**")

    if parts:
        content = "Here's what I remember about you:\n" + "\n".join(parts)
    else:
        content = (
            "I don't have any saved travel preferences yet. "
            "Tell me your preferred airline, dietary needs, number of travelers, "
            "or other travel preferences."
        )

    logger.info("Preferences memory agent answered from saved state.")
    return {"messages": [AIMessage(content=content)]}


def _update_preferences(state: AgentState, raw_message: str) -> dict:
    """
    Extracts travel preferences from the latest user message and persists them
    into the graph state.
    """
    message = raw_message.lower()
    updates: dict = {}

    airline = _extract_airline(message)
    if airline:
        updates["preferred_airline"] = airline
        logger.info("Preference detected — airline: %s", airline)

    food = _extract_food_preference(message)
    if food:
        updates["food_preference"] = food
        logger.info("Preference detected — food: %s", food)

    travelers = _extract_num_travelers(message)
    if travelers is not None:
        updates["num_travelers"] = travelers
        logger.info("Preference detected — travelers: %s", travelers)

    extra = _extract_preference_with_llm(raw_message)
    if extra:
        existing = state.get("travel_preferences", "") or ""
        if extra.lower() not in existing.lower():
            updates["travel_preferences"] = (
                f"{existing}\n- {extra}".strip() if existing else f"- {extra}"
            )
            logger.info("Preference detected by LLM — %s", extra)

    updates["messages"] = [AIMessage(content=_build_ack_message(updates))]

    logger.info("Preferences memory agent updated %d fields.", len(updates) - 1)
    return updates


def _extract_airline(message: str) -> Optional[str]:
    """
    Extracts a known preferred airline from the message.
    """
    for airline in PREFERENCE_EXTRACTION_CONFIG.airline_names:
        if airline in message:
            return airline.title()

    return None


def _extract_food_preference(message: str) -> Optional[str]:
    """
    Extracts a known dietary preference from the message.
    """
    for food in PREFERENCE_EXTRACTION_CONFIG.food_preferences:
        if food in message:
            return "gluten-free" if food == "gluten free" else food

    return None


def _extract_num_travelers(message: str) -> Optional[int]:
    """
    Extracts the number of travelers from common travel preference phrases.
    """
    patterns = [
        r"(\d+)\s*(people|person|traveler|travelers|traveller|travellers|passenger|passengers|of us|pax)",
        r"we\s+are\s+(\d+)",
        r"family\s+of\s+(\d+)",
        r"travelling\s+with\s+(\d+)",
        r"traveling\s+with\s+(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return int(match.group(1))

    return None


def _extract_preference_with_llm(message: str) -> Optional[str]:
    """
    Uses Groq llama-3.1-8b-instant to extract additional travel preferences
    that are not covered by deterministic airline, food, and traveler detection.

    Returns a short normalized phrase or None.
    """
    if not os.getenv("GROQ_API_KEY", "").startswith("gsk_"):
        return None

    try:
        from langchain_groq import ChatGroq

        model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=30,
            timeout=5,
        )

        response = model.invoke([
            SystemMessage(content=(
                "You extract specific travel preferences from user messages.\n"
                "Return ONLY a short phrase, max 8 words, or exactly 'none'.\n"
                "Skip airline names, food types, and number of travelers because those are handled elsewhere.\n\n"
                "Examples:\n"
                "'I prefer Airbus planes for safety' -> 'Prefers Airbus aircraft'\n"
                "'I want window seats always' -> 'Prefers window seats'\n"
                "'I only take direct flights' -> 'Direct flights only'\n"
                "'I like morning departures' -> 'Prefers morning departures'\n"
                "'I need wheelchair accessibility' -> 'Requires wheelchair access'\n"
                "'I prefer business class' -> 'Prefers business class'\n"
                "'I prefer 5-star hotels' -> 'Prefers 5-star hotels'\n"
                "'I prefer El Al and kosher' -> 'none'\n"
                "'Plan a trip to Paris' -> 'none'\n"
                "'I travel with 2 people' -> 'none'"
            )),
            HumanMessage(content=f"Message: \"{message}\"\nPreference:"),
        ])

        result = response.content.strip().strip("\"'").strip()
        if not result or result.lower() == "none":
            return None

        return result

    except Exception as error:
        logger.warning("Preference LLM extraction failed: %s", error)
        return None


def _build_ack_message(updates: dict) -> str:
    """
    Builds a user-facing acknowledgement for saved preference updates.
    """
    saved = []

    if updates.get("preferred_airline"):
        saved.append(f"Airline: **{updates['preferred_airline']}**")

    if updates.get("food_preference"):
        saved.append(f"Food: **{updates['food_preference']}**")

    if updates.get("num_travelers"):
        saved.append(f"Travelers: **{updates['num_travelers']}**")

    if updates.get("travel_preferences"):
        last_extra = updates["travel_preferences"].strip().splitlines()[-1].lstrip("- ")
        saved.append(f"Additional: **{last_extra}**")

    if not saved:
        return (
            "I understood this as a preferences update, but I could not detect a specific "
            "travel preference. Try mentioning your preferred airline, food preference, "
            "number of travelers, seat preference, hotel preference, or flight style."
        )

    return "Got it! I've saved your travel preferences:\n" + "\n".join(
        f"- {item}" for item in saved
    )