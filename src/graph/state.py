from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Central shared memory across all graph nodes.

    messages               — full conversation history (append-only via add_messages reducer)
    current_city           — destination extracted from the latest user message
    total_budget           — budget extracted from the latest user message (USD)
    tool_call_count        — incremented each time the agent requests a tool; guards loops
    validation_status      — set by validator: "approved" | "blocked_injection" |
                             "blocked_scope" | "blocked_city" | "blocked_harm"
    orchestrator_route     — selected by master_orchestrator:
                             "preferences_memory" | "research" | "cache_check"
    orchestrator_reason    — short explanation for the selected orchestrator route
    preferred_airline      — user's preferred airline, persisted across sessions
    food_preference        — user's dietary preference (kosher, vegan …), persisted
    num_travelers          — number of people traveling, persisted across sessions
    is_admin               — True when session ID ends with ADMIN00; controls reviewer routing
    conversation_summary   — compact summary of past messages produced by summarizer_node
    travel_preferences     — additional free-form travel preferences persisted across sessions
    cache_status           — set by cache checker: "hit" | "miss"
    cache_similarity_score — best semantic similarity score found by cache checker
    cache_matched_query    — cached query that best matched the current query
    cache_answer           — cached answer returned on cache hit
    """

    messages: Annotated[list, add_messages]
    current_city: str
    total_budget: float
    tool_call_count: int
    validation_status: str
    orchestrator_route: str
    orchestrator_reason: str
    preferred_airline: str
    food_preference: str
    num_travelers: int
    is_admin: bool
    conversation_summary: str
    travel_preferences: str
    cache_status: str
    cache_similarity_score: float
    cache_matched_query: str
    cache_answer: str