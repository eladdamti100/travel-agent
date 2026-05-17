from dataclasses import dataclass


@dataclass(frozen=True)
class PreferenceExtractionConfig:
    """
    Static configuration used by the preferences memory agent.

    These values are intentionally kept outside the agent so the agent contains
    behavior, while this module contains domain vocabulary and trigger phrases.
    """

    airline_names: tuple[str, ...]
    food_preferences: tuple[str, ...]
    update_triggers: tuple[str, ...]
    recall_triggers: tuple[str, ...]


PREFERENCE_EXTRACTION_CONFIG = PreferenceExtractionConfig(
    airline_names=(
        "el al",
        "emirates",
        "lufthansa",
        "british airways",
        "air france",
        "united",
        "virgin atlantic",
        "ryanair",
        "turkish airlines",
        "klm",
        "swiss",
        "tap",
        "wizz",
        "easyjet",
    ),
    food_preferences=(
        "kosher",
        "vegan",
        "vegetarian",
        "halal",
        "gluten-free",
        "gluten free",
    ),
    update_triggers=(
        "i prefer",
        "i like",
        "my favourite",
        "my favorite",
        "always fly",
        "always travel with",
        "i'm traveling with",
        "i am traveling with",
        "traveling with",
        "travelling with",
        "people traveling",
        "people travelling",
        "we are",
        "family of",
        "kosher",
        "vegan",
        "vegetarian",
        "halal",
        "gluten",
    ),
    recall_triggers=(
        "what do i prefer",
        "what's my preference",
        "what are my preferences",
        "do you remember",
        "what airline do i",
        "my preference",
        "what food",
        "remember my",
        "what did i tell you",
        "my saved",
        "my profile",
        "what do you know about me",
        "my details",
    ),
)