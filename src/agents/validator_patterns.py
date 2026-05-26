"""
Regex patterns and keyword sets for the hardcoded input validator.
"""

SUPPORTED_CITIES = frozenset({"paris", "london", "tokyo", "new york", "berlin"})

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

HARM_PATTERNS_RAW = [
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

INJECTION_PATTERNS_RAW = [
    r"ignore\s+(all\s+)?(previous\s+|your\s+|the\s+|above\s+)?(instructions?|prompts?|commands?|rules?|context)",
    r"disregard\s+(all\s+)?(previous\s+|your\s+|the\s+)?(instructions?|prompts?|commands?|rules?|context)",
    r"follow\s+(the\s+)?(next|this|my|new|following)\s+(command|instruction|prompt|rule|order)",
    r"answer\s+(me\s+)?only\s+(in|using|with|like)\s+\w+",
    r"respond\s+only\s+(in|using|with|like)\s+\w+",
    r"(speak|write|talk|communicate|reply)\s+only\s+(in|using|like|as)\s+\w+",
    r"change\s+(your\s+)?(tone|style|language|personality|character|voice|way\s+of)",
    r"(from\s+now\s+on|starting\s+now|henceforth)\s+.*(speak|respond|answer|write|talk)",
    r"new\s+system\s+prompt",
    r"your\s+new\s+role",
    r"\bact\s+as\s+(a|an|if)\b",
    r"pretend\s+(you\s+are|to\s+be)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"you\s+are\s+(actually|really|secretly|truly)\s+a",
    r"(switch|change|enter)\s+(to\s+)?(a\s+)?(different|new|unrestricted)\s+mode",
    r"simulate\s+(a|an)\s+.*(ai|assistant|bot|system)",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"what\s+are\s+your\s+(instructions?|rules?|prompt|system)",
    r"(repeat|show|reveal|print|output)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
    r"forget\s+(everything|all|your\s+(instructions?|rules?|prompts?))",
    r"override\s+(your|the)\s+(instructions?|prompt|rules?|system)",
    r"(developer|god|admin|sudo|root)\s+mode",
    r"bypass\s+(your|the)\s+(restrictions?|rules?|guidelines?|filters?|safety)",
    r"from\s+now\s+on\s+(you|act|be|ignore|forget)",
]

OFF_TOPIC_PATTERNS_RAW = [
    (r"\b(solve|calculate|compute|evaluate)\s+(this\s+)?(equation|math|formula|integral|derivative|sum|problem)", "mathematics"),
    (r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b", "arithmetic"),
    (r"\bwrite\s+(me\s+)?(a\s+)?(poem|essay|story|song|lyrics|novel|script|haiku|sonnet)", "creative writing"),
    (r"\b(write|generate|create|give\s+me)\s+(\w+\s+){0,4}(code|function|class|algorithm|script|program)\b", "coding"),
    (r"\b(debug|fix|review)\s+(this|my|the)\s+(code|function|script|program|bug)", "coding"),
    (r"\bhow\s+(do\s+i|to|can\s+i)\s+(reverse|sort|search|traverse|implement|merge|split|flatten|parse|serialize)\s+(a\s+)?(linked\s+list|array|string|tree|graph|stack|queue|dict|list|tuple)", "coding"),
    (r"\bhow\s+(do\s+i|to|can\s+i)\s+(write|code|build|create|make|implement)\s+(a\s+)?(function|class|loop|recursion|algorithm|api|server|database|query)", "coding"),
    (r"\b(linked\s+list|binary\s+tree|binary\s+search|hash\s+table|hash\s+map|depth.first|breadth.first|big.o\s+notation|time\s+complexity|space\s+complexity)\b", "computer science"),
    (r"\b(recursion|polymorphism|inheritance|encapsulation|abstraction|object.oriented|functional\s+programming)\b", "computer science"),
    (r"\bin\s+(python|java(?:script)?|c\+\+|c#|ruby|golang|go|rust|php|swift|kotlin|typescript|scala|r\b)\b", "programming language"),
    (r"\b(def\s+\w+|class\s+\w+|import\s+\w+|print\s*\(|console\.log|System\.out)\b", "code snippet"),
    (r"\bwhat\s+is\s+the\s+(capital|population|president|prime\s+minister|gdp|area)\s+of\b", "general knowledge"),
    (r"\bwho\s+(is|was|invented|discovered|wrote|created|founded)\b", "general knowledge"),
    (r"\bexplain\s+(to\s+me\s+)?(what|how|why)\s+(is|are|does|do)\s+(machine\s+learning|deep\s+learning|neural|quantum|blockchain|ai|llm)\b", "general knowledge"),
    (r"\btranslate\s+(this|the|from|to|into)\b", "translation"),
    (r"\bplay\s+(a\s+)?(game|chess|quiz|trivia|riddle)\b", "games"),
    (r"\b(stock|crypto|bitcoin|ethereum|forex)\s+(market|price|trading|chart)\b", "finance"),
    (r"\b(recipe|how\s+to\s+cook|how\s+to\s+bake|ingredient|dish)\b", "cooking"),
    (r"\bsport(s)?\s+(score|result|match|standings|league)\b", "sports"),
    (r"\b(diagnosis|symptom|medicine|prescription|disease|treatment)\b", "medical"),
    (r"\b(law|legal\s+advice|is\s+it\s+legal|lawsuit|attorney)\b", "legal"),
    (r"\bthe\s+(meaning\s+of\s+life|universe|everything|big\s+bang)\b", "philosophy/science"),
    (r"\b(weather|temperature|forecast|rain|snow|sunny|cloudy|humidity)\s+(in|at|for|today|tomorrow|right\s+now)\b", "weather"),
    (r"\bwhat('s|\s+is)\s+the\s+weather\b", "weather"),
    (r"\btell\s+me\s+a\s+(joke|riddle|pun|story)\b", "entertainment"),
    (r"\b(make\s+me\s+laugh|say\s+something\s+funny)\b", "entertainment"),
    (r"\b(latest|breaking|today'?s?)\s+(news|headlines)\b", "news"),
    (r"\bwhat('s|\s+is)\s+(happening|in\s+the\s+news)\b", "news"),
    (r"\b(tinder|instagram|snapchat|tiktok|dating\s+app|how\s+to\s+get\s+a\s+(girlfriend|boyfriend|date))\b", "social"),
]

STRONG_TRAVEL_SIGNALS = frozenset({
    "flight", "flights", "hotel", "hotels", "itinerary",
    "visa", "airport", "airline", "airlines", "vacation", "holiday",
    "accommodation", "ticket", "passport", "sightseeing",
    "travel", "travelling", "traveling", "trip", "fly", "flying",
    "tourism", "tourist", "tour", "destination",
})

TRAVEL_KEYWORDS = frozenset({
    "flight", "flights", "hotel", "hotels", "trip", "travel", "travelling",
    "destination", "visa", "activities", "activity", "itinerary", "airport",
    "airline", "vacation", "holiday", "tourism", "tourist", "accommodation",
    "booking", "book", "ticket", "passport", "tour", "sightseeing",
    "cheapest", "budget", "cost", "price", "nights", "days",
    "plan", "help", "hi", "hello",
    "prefer", "preference", "favourite", "favorite", "kosher", "vegan",
    "vegetarian", "halal", "traveler", "travellers", "travelers", "passenger",
    "flying", "fly",
    "japan", "france", "germany",
    "uk", "england", "britain", "united kingdom",
    "usa", "america", "united states",
})

VISA_COUNTRY_NAMES = frozenset({
    "japan", "france", "germany",
    "uk", "england", "britain", "united kingdom",
    "usa", "america", "united states",
})
