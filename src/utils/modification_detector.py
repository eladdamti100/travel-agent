"""
Detects when a user is modifying trip parameters rather than planning a new trip.

When modifications are detected, the cache should be bypassed and re-planning triggered.
"""

import re
from typing import Optional


# Keywords that indicate a parameter modification is happening
MODIFICATION_KEYWORDS = {
    # Change/modify verbs
    "change", "update", "modify", "replace", "switch", "different", "new",
    "instead", "other", "another", "from", "to",
    # Negative contrast
    "not", "no", "without",
    # Specification keywords
    "rather", "prefer", "want",
}

# Trip parameters that can be modified
TRIP_PARAMETERS = {
    "airport", "origin", "departure", "starting", "leaving from",
    "destination", "arriving at", "going to",
    "budget", "cost", "price", "spending",
    "days", "nights", "duration", "length", "week", "month",
    "travelers", "people", "passengers", "group",
    "hotel", "accommodation", "stay", "place",
    "flight", "flights", "airline", "carrier",
    "activity", "activities", "things to do", "attractions",
    "date", "dates", "when", "departure date", "arrival date",
    "visa", "passport",
}


def detect_modification_context(user_message: str) -> bool:
    """
    Returns True if the user message appears to be modifying a trip parameter
    rather than planning a new trip.

    Patterns detected:
    - "change X to Y"
    - "instead of X, use Y"
    - "different airport"
    - "not JFK, but TLV"
    - "from JFK to TLV"
    """
    if not user_message:
        return False

    lower_msg = user_message.lower()

    # Check for explicit modification verbs + parameter words
    # Examples: "change airport", "update budget", "switch hotel"
    for verb in ["change", "update", "modify", "replace", "switch"]:
        for param in TRIP_PARAMETERS:
            pattern = rf"\b{verb}\s+.*\b{param}s?\b"
            if re.search(pattern, lower_msg):
                return True

    # Check for "instead of" / "instead" patterns
    if "instead of" in lower_msg or "rather than" in lower_msg:
        for param in TRIP_PARAMETERS:
            if param in lower_msg:
                return True
        # "instead of JFK, TLV" — two airport codes with "instead of"
        import re as _re
        if _re.search(r"\b[A-Z]{3}\b", lower_msg.upper()):
            return True

    if "instead" in lower_msg:
        for param in TRIP_PARAMETERS:
            if param in lower_msg:
                return True

    # Check for contrasting patterns: "not X, Y" or "X to Y"
    # Examples: "not JFK, TLV" or "from JFK to TLV"
    if re.search(r"\b(not|no)\s+\w+\s*,", lower_msg):
        for param in TRIP_PARAMETERS:
            if param in lower_msg:
                return True

    # Check for "from X to Y" with two IATA airport codes — indicates a route swap.
    # Requires explicit modification verbs ("change", "switch") OR two airport codes
    # in a from→to pattern to avoid false-positives on "flying from TLV to Paris".
    if " to " in lower_msg:
        airport_codes = re.findall(r"\b([A-Z]{3})\b", user_message)
        has_two_airports = len(set(airport_codes)) >= 2
        has_mod_verb = any(w in lower_msg for w in ["change", "switch", "different", "instead"])
        from_to_swap = re.search(r"\bfrom\s+[A-Z]{3}\b.*\bto\s+[A-Z]{3}\b", user_message)
        if (has_two_airports and from_to_swap) or has_mod_verb:
            if re.search(r"\b[A-Z]{3}\b", user_message):
                return True

    # Check for "different" + parameter
    if "different" in lower_msg or "another" in lower_msg:
        for param in TRIP_PARAMETERS:
            if param in lower_msg:
                return True

    return False


def extract_modified_parameters(user_message: str) -> dict:
    """
    Extracts what parameters the user is trying to modify.

    Returns a dict with keys like:
    - origin_airport: new origin airport code if detected
    - destination: new destination if detected
    - budget: new budget if detected
    - duration: new duration if detected
    """
    params = {}
    lower_msg = user_message.lower()

    # Extract airport codes (3-letter IATA codes)
    airport_codes = re.findall(r"\b([A-Z]{3})\b", user_message)
    if airport_codes and any(word in lower_msg for word in ["change", "from", "airport"]):
        if len(airport_codes) >= 1:
            params["origin_airport"] = airport_codes[0]
        if len(airport_codes) >= 2:
            params["origin_airport"] = airport_codes[0]

    # Extract budget (looking for $amount patterns)
    budget_match = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", user_message)
    if budget_match and any(word in lower_msg for word in ["budget", "cost", "price", "spending"]):
        budget = float(budget_match.group(1).replace(",", ""))
        params["budget"] = budget

    # Extract duration
    duration_match = re.search(r"(\d+)\s*(days?|nights?|weeks?)", lower_msg)
    if duration_match and any(word in lower_msg for word in ["days", "nights", "duration", "length"]):
        params["duration"] = f"{duration_match.group(1)} {duration_match.group(2)}"

    return params
