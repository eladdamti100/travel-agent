"""
Validator — hardcoded security guardrail, zero LLM calls.

Checks every user message for four threat categories before Marco runs:
  0. Harmful content    — violence, illegal requests, hate speech, abuse
  1. Prompt injection   — attempts to override system instructions
  2. Out-of-scope       — requests completely unrelated to travel
  3. Unsupported city   — destination not in the database

All checks are regex / keyword based and compile once at class load time.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    approved: bool
    verdict: str          # APPROVED | BLOCKED_HARM | BLOCKED_INJECTION | BLOCKED_SCOPE | BLOCKED_CITY
    reason: str
    rejection_message: str


# ── Rejection messages shown to the user ─────────────────────────────────────

_REJECTION_MESSAGES = {
    "BLOCKED_HARM": (
        "I'm sorry, but I'm unable to process your request as it appears to violate "
        "our usage guidelines. I'm Marco, a travel planning assistant here to help you "
        "plan wonderful trips. Please keep our conversation respectful and travel-related. "
        "Feel free to ask me about flights, hotels, or activities!"
    ),
    "BLOCKED_INJECTION": (
        "I noticed your message contains instructions trying to change my behaviour. "
        "I'm Marco, your travel planning assistant, and I can only help with "
        "flights, hotels, activities, and trip planning."
    ),
    "BLOCKED_SCOPE": (
        "That's outside what I can help with. I'm Marco, a travel planning assistant. "
        "Ask me about flights, hotels, activities, or trip costs for "
        "Paris, London, Tokyo, New York, or Berlin."
    ),
    "BLOCKED_CITY": (
        "Sorry, that destination isn't in my database yet. "
        "I currently support: Paris, London, Tokyo, New York, and Berlin. "
        "Would you like to plan a trip to one of those instead?"
    ),
}


# ── Main validator class ──────────────────────────────────────────────────────

class InputValidator:
    """
    Stateless hardcoded validator.  All patterns compile once on first use.
    Call InputValidator.validate(message) → ValidationResult.
    detect_harm() is a public static method and can be called independently.
    """

    # Cities in the database
    SUPPORTED_CITIES = frozenset({"paris", "london", "tokyo", "new york", "berlin"})

    # Common world cities NOT in the database — detected to give a helpful block
    KNOWN_UNSUPPORTED_CITIES = frozenset({
        "rome", "madrid", "amsterdam", "dubai", "bangkok", "sydney",
        "barcelona", "singapore", "istanbul", "prague", "vienna",
        "los angeles", "chicago", "toronto", "seoul", "beijing", "shanghai",
        "hong kong", "mumbai", "delhi", "cairo", "mexico city",
        "buenos aires", "johannesburg", "moscow", "athens", "lisbon",
        "florence", "venice", "milan", "brussels", "geneva", "zurich",
        "stockholm", "oslo", "copenhagen", "helsinki", "warsaw", "budapest",
        "kyoto", "osaka", "bali", "phuket", "cancun", "havana",
        "nairobi", "casablanca", "abu dhabi", "doha", "riyadh",
        "tel aviv", "jerusalem", "beirut", "karachi", "lahore",
        "lagos", "accra", "tunis", "algiers", "cape town",
    })

    # ── Harm patterns (checked before everything else) ────────────────────────
    _HARM_PATTERNS_RAW = [
        (r"\b(kill|murder|shoot|stab|blow\s+up|slaughter)\s+(someone|people|person|him|her|them|you|us|everyone)\b", "threat of violence"),
        (r"\bi\s+(want\s+to|will|am\s+going\s+to)\s+(kill|murder|hurt|attack|destroy|harm)\b", "direct threat"),
        (r"\bhow\s+to\s+(make|build|create|synthesize)\s+(a\s+)?(bomb|weapon|explosive|poison|bioweapon|nerve\s+agent|drug)\b", "dangerous instructions"),
        (r"\b(suicide|self[\-\s]harm|kill\s+myself|end\s+my\s+life|want\s+to\s+die)\b", "self-harm"),
        (r"\b(hack|breach|crack)\s+(into\s+)?(the\s+)?(system|server|database|account|network|mainframe)\b", "hacking"),
        (r"\b(child\s+porn|csam|underage\s+(sex|nude|naked|porn))\b", "csam"),
        (r"\b(ethnic\s+cleansing|genocide|hate\s+crime|racial\s+violence)\b", "hate crime"),
        (r"\b(fuck\s+you|go\s+to\s+hell|you\s+(suck|are\s+stupid|idiot|moron))\b", "abusive language"),
        (r"\b(steal|rob|defraud|scam|phish)\s+(credit\s+card|identity|money|bank|people)\b", "fraud"),
        (r"\b(drug\s+deal|sell\s+drugs|buy\s+cocaine|buy\s+heroin|smuggl(e|ing))\b", "illegal substances"),
    ]

    # Regex patterns that signal prompt injection
    _INJECTION_PATTERNS_RAW = [
        # ignore / disregard — catches any combination of modifiers before the target word
        r"ignore\s+(all\s+)?(previous\s+|your\s+|the\s+|above\s+)?(instructions?|prompts?|commands?|rules?|context)",
        r"disregard\s+(all\s+)?(previous\s+|your\s+|the\s+)?(instructions?|prompts?|commands?|rules?|context)",

        # "follow the next command / instruction / prompt"
        r"follow\s+(the\s+)?(next|this|my|new|following)\s+(command|instruction|prompt|rule|order)",

        # Style / personality change requests
        r"answer\s+(me\s+)?(only\s+)?(in|using|with|like)\s+\w+",
        r"respond\s+(only\s+)?(in|using|with|like)\s+\w+",
        r"(speak|write|talk|communicate|reply)\s+(only\s+)?(in|using|like|as)\s+\w+",
        r"change\s+(your\s+)?(tone|style|language|personality|character|voice|way\s+of)",
        r"(from\s+now\s+on|starting\s+now|henceforth)\s+.*(speak|respond|answer|write|talk)",

        # Role / identity injection
        r"new\s+system\s+prompt",
        r"your\s+new\s+role",
        r"\bact\s+as\s+(a|an|if)\b",
        r"pretend\s+(you\s+are|to\s+be)",
        r"you\s+are\s+now\s+(a|an)\b",
        r"you\s+are\s+(actually|really|secretly|truly)\s+a",
        r"(switch|change|enter)\s+(to\s+)?(a\s+)?(different|new|unrestricted)\s+mode",
        r"simulate\s+(a|an)\s+.*(ai|assistant|bot|system)",

        # Jailbreak keywords
        r"\bjailbreak\b",
        r"\bDAN\b",
        r"do\s+anything\s+now",

        # Reveal / override system
        r"what\s+are\s+your\s+(instructions?|rules?|prompt|system)",
        r"(repeat|show|reveal|print|output)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
        r"forget\s+(everything|all|your\s+(instructions?|rules?|prompts?))",
        r"override\s+(your|the)\s+(instructions?|prompt|rules?|system)",
        r"(developer|god|admin|sudo|root)\s+mode",
        r"bypass\s+(your|the)\s+(restrictions?|rules?|guidelines?|filters?|safety)",
        r"from\s+now\s+on\s+(you|act|be|ignore|forget)",
    ]

    # (pattern, topic_label) pairs for clearly off-topic content
    _OFF_TOPIC_PATTERNS_RAW = [
        # Mathematics
        (r"\b(solve|calculate|compute|evaluate)\s+(this\s+)?(equation|math|formula|integral|derivative|sum|problem)", "mathematics"),
        (r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b", "arithmetic"),

        # Creative writing
        (r"\bwrite\s+(me\s+)?(a\s+)?(poem|essay|story|song|lyrics|novel|script|haiku|sonnet)", "creative writing"),

        # Coding — explicit write/create requests (flexible middle words: "write me a Python function")
        (r"\b(write|generate|create|give\s+me)\s+(\w+\s+){0,4}(code|function|class|algorithm|script|program)\b", "coding"),
        (r"\b(debug|fix|review)\s+(this|my|the)\s+(code|function|script|program|bug)", "coding"),

        # Coding — "how to" programming questions
        (r"\bhow\s+(do\s+i|to|can\s+i)\s+(reverse|sort|search|traverse|implement|merge|split|flatten|parse|serialize)\s+(a\s+)?(linked\s+list|array|string|tree|graph|stack|queue|dict|list|tuple)", "coding"),
        (r"\bhow\s+(do\s+i|to|can\s+i)\s+(write|code|build|create|make|implement)\s+(a\s+)?(function|class|loop|recursion|algorithm|api|server|database|query)", "coding"),

        # Programming data structures and concepts (clearly non-travel)
        (r"\b(linked\s+list|binary\s+tree|binary\s+search|hash\s+table|hash\s+map|depth.first|breadth.first|big.o\s+notation|time\s+complexity|space\s+complexity)\b", "computer science"),
        (r"\b(recursion|polymorphism|inheritance|encapsulation|abstraction|object.oriented|functional\s+programming)\b", "computer science"),

        # Programming language syntax
        (r"\bin\s+(python|java(?:script)?|c\+\+|c#|ruby|golang|go|rust|php|swift|kotlin|typescript|scala|r\b)\b", "programming language"),
        (r"\b(def\s+\w+|class\s+\w+|import\s+\w+|print\s*\(|console\.log|System\.out)\b", "code snippet"),

        # General knowledge
        (r"\bwhat\s+is\s+the\s+(capital|population|president|prime\s+minister|gdp|area)\s+of\b", "general knowledge"),
        (r"\bwho\s+(is|was|invented|discovered|wrote|created|founded)\b", "general knowledge"),
        (r"\bexplain\s+(to\s+me\s+)?(what|how|why)\s+(is|are|does|do)\s+(machine\s+learning|deep\s+learning|neural|quantum|blockchain|ai|llm)\b", "general knowledge"),

        # Translation
        (r"\btranslate\s+(this|the|from|to|into)\b", "translation"),

        # Games
        (r"\bplay\s+(a\s+)?(game|chess|quiz|trivia|riddle)\b", "games"),

        # Finance
        (r"\b(stock|crypto|bitcoin|ethereum|forex)\s+(market|price|trading|chart)\b", "finance"),

        # Cooking
        (r"\b(recipe|how\s+to\s+cook|how\s+to\s+bake|ingredient|dish)\b", "cooking"),

        # Sports
        (r"\bsport(s)?\s+(score|result|match|standings|league)\b", "sports"),

        # Medical
        (r"\b(diagnosis|symptom|medicine|prescription|disease|treatment)\b", "medical"),

        # Legal
        (r"\b(law|legal\s+advice|is\s+it\s+legal|lawsuit|attorney)\b", "legal"),

        # Philosophy/science
        (r"\bthe\s+(meaning\s+of\s+life|universe|everything|big\s+bang)\b", "philosophy/science"),

        # Weather (the original bug)
        (r"\b(weather|temperature|forecast|rain|snow|sunny|cloudy|humidity)\s+(in|at|for|today|tomorrow|right\s+now)\b", "weather"),
        (r"\bwhat('s|\s+is)\s+the\s+weather\b", "weather"),

        # Entertainment
        (r"\btell\s+me\s+a\s+(joke|riddle|pun|story)\b", "entertainment"),
        (r"\b(make\s+me\s+laugh|say\s+something\s+funny)\b", "entertainment"),

        # News
        (r"\b(latest|breaking|today'?s?)\s+(news|headlines)\b", "news"),
        (r"\bwhat('s|\s+is)\s+(happening|in\s+the\s+news)\b", "news"),

        # Social media / relationships
        (r"\b(tinder|instagram|snapchat|tiktok|dating\s+app|how\s+to\s+get\s+a\s+(girlfriend|boyfriend|date))\b", "social"),
    ]

    # Travel-related keywords — any match overrides the off-topic check
    _TRAVEL_KEYWORDS = frozenset({
        "flight", "flights", "hotel", "hotels", "trip", "travel", "travelling",
        "destination", "visa", "activities", "activity", "itinerary", "airport",
        "airline", "vacation", "holiday", "tourism", "tourist", "accommodation",
        "booking", "book", "ticket", "passport", "tour", "sightseeing",
        "cheapest", "budget", "cost", "price", "nights", "days",
        "plan", "help", "hi", "hello",
        # Preference keywords — always travel-related
        "prefer", "preference", "favourite", "favorite", "kosher", "vegan",
        "vegetarian", "halal", "traveler", "travellers", "travelers", "passenger",
        "flying", "fly",
        # Country names for visa checks — map to supported cities
        # Japan→Tokyo, France→Paris, uk/england/britain→London,
        # germany→Berlin, usa/america→New York
        "japan", "france", "germany",
        "uk", "england", "britain", "united kingdom",
        "usa", "america", "united states",
    })

    # Country names that correspond to supported cities — never block these
    # even though they are not city names in SUPPORTED_CITIES
    _VISA_COUNTRY_NAMES = frozenset({
        "japan", "france", "germany",
        "uk", "england", "britain", "united kingdom",
        "usa", "america", "united states",
    })

    # Compiled pattern caches
    _harm_compiled: Optional[list] = None
    _injection_compiled: Optional[list] = None
    _off_topic_compiled: Optional[list] = None

    @classmethod
    def _harm_patterns(cls) -> list:
        if cls._harm_compiled is None:
            cls._harm_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._HARM_PATTERNS_RAW
            ]
        return cls._harm_compiled

    @classmethod
    def _injection_patterns(cls) -> list:
        if cls._injection_compiled is None:
            cls._injection_compiled = [
                re.compile(p, re.IGNORECASE) for p in cls._INJECTION_PATTERNS_RAW
            ]
        return cls._injection_compiled

    @classmethod
    def _off_topic_patterns(cls) -> list:
        if cls._off_topic_compiled is None:
            cls._off_topic_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._OFF_TOPIC_PATTERNS_RAW
            ]
        return cls._off_topic_compiled

    @staticmethod
    def detect_harm(message: str) -> Optional[tuple]:
        """
        Public static method — scans message for harmful or violating content.

        Checks for: violence, dangerous instructions, self-harm, hacking,
        hate crimes, abusive language, fraud, and illegal activity.

        Returns (verdict, reason) tuple if harm is detected, None if clean.
        Can be called independently: InputValidator.detect_harm(message)

        When harm is detected the caller should cancel all further processing
        and return the BLOCKED_HARM rejection message to the user.
        """
        for pattern, label in InputValidator._harm_patterns():
            if pattern.search(message):
                return ("BLOCKED_HARM", f"Harmful content detected: {label}")
        return None

    @classmethod
    def validate(cls, message: str) -> ValidationResult:
        msg_lower = message.lower().strip()

        # ── 0. Harm check (absolute highest priority) ─────────────────────────
        harm = cls.detect_harm(message)
        if harm:
            verdict, reason = harm
            return ValidationResult(
                approved=False,
                verdict=verdict,
                reason=reason,
                rejection_message=_REJECTION_MESSAGES["BLOCKED_HARM"],
            )

        # ── 1. Injection check ────────────────────────────────────────────────
        for pattern in cls._injection_patterns():
            if pattern.search(message):
                return ValidationResult(
                    approved=False,
                    verdict="BLOCKED_INJECTION",
                    reason="Prompt injection attempt detected.",
                    rejection_message=_REJECTION_MESSAGES["BLOCKED_INJECTION"],
                )

        # ── 2. City check ─────────────────────────────────────────────────────
        detected = cls._detect_city(msg_lower)
        if detected and detected not in cls.SUPPORTED_CITIES:
            return ValidationResult(
                approved=False,
                verdict="BLOCKED_CITY",
                reason=f"Unsupported destination: {detected}.",
                rejection_message=(
                    f"Sorry, {detected.title()} isn't in my database yet. "
                    "I currently support: Paris, London, Tokyo, New York, and Berlin. "
                    "Would you like to plan a trip to one of those instead?"
                ),
            )

        # ── 3. Off-topic check (only if no travel keyword present) ────────────
        has_travel_keyword = any(kw in msg_lower for kw in cls._TRAVEL_KEYWORDS)
        if not has_travel_keyword:
            for pattern, topic in cls._off_topic_patterns():
                if pattern.search(message):
                    return ValidationResult(
                        approved=False,
                        verdict="BLOCKED_SCOPE",
                        reason=f"Off-topic request: {topic}.",
                        rejection_message=_REJECTION_MESSAGES["BLOCKED_SCOPE"],
                    )

        # ── 4. Approved ───────────────────────────────────────────────────────
        return ValidationResult(
            approved=True,
            verdict="APPROVED",
            reason="Valid travel-related request.",
            rejection_message="",
        )

    @classmethod
    def is_clearly_travel(cls, message: str) -> bool:
        """
        Returns True when the message unambiguously relates to travel.

        Used by nodes.py to skip the LLM validator and fast-approve obvious requests,
        saving ~200ms per typical travel message.
        """
        msg_lower = message.lower()
        return any(kw in msg_lower for kw in cls._TRAVEL_KEYWORDS)

    @classmethod
    def _detect_city(cls, msg_lower: str) -> Optional[str]:
        for city in cls.SUPPORTED_CITIES:
            if city in msg_lower:
                return city
        # Country names that map to supported cities are never unsupported
        for country in cls._VISA_COUNTRY_NAMES:
            if country in msg_lower:
                return None
        for city in cls.KNOWN_UNSUPPORTED_CITIES:
            if city in msg_lower:
                return city
        return None


# ── Public function (called by nodes.py) ─────────────────────────────────────

def validate_input(user_message: str) -> ValidationResult:
    return InputValidator.validate(user_message)
