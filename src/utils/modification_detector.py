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
            pattern = rf"\b{verb}\s+.*\b{param}\b"
            if re.search(pattern, lower_msg):
                return True

    # Check for "instead of" patterns
    if "instead of" in lower_msg or "rather than" in lower_msg:
        for param in TRIP_PARAMETERS:
            if param in lower_msg:
                return True

    # Check for contrasting patterns: "not X, Y" or "X to Y"
    # Examples: "not JFK, TLV" or "from JFK to TLV"
    if re.search(r"\b(not|no)\s+\w+\s*,", lower_msg):
        for param in TRIP_PARAMETERS:
            if param in lower_msg:
                return True

    # Check for "to" with airport/destination keywords
    if " to " in lower_msg:
        # Only match if it looks like a modification (has airport code or known airport)
        airport_pattern = r"\b[A-Z]{3}\b"  # 3-letter IATA code
        if re.search(airport_pattern, user_message):
            if any(word in lower_msg for word in ["change", "from", "different", "instead", "switch"]):
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
