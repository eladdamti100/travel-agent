from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Central shared memory across all graph nodes.

    messages               — full conversation history
    current_city           — destination extracted from the latest user message
    total_budget           — budget extracted from the latest user message
    tool_call_count        — incremented each time the agent requests a tool
    validation_status      — set by validator
    orchestrator_route     — "preferences_memory" | "research" | "cache_check"
    orchestrator_reason    — short explanation for the selected route

    preferred_airline      — persisted airline preference
    food_preference        — persisted dietary preference
    num_travelers          — persisted/default number of travelers
    travel_preferences     — additional persisted free-form travel preferences

    is_admin               — True when session ID ends with ADMIN00
    conversation_summary   — compact summary of past messages

    cache_status           — "hit" | "miss"
    cache_similarity_score — best semantic similarity score
    cache_matched_query    — cached query that best matched the current query
    cache_answer           — cached answer returned on cache hit

    trip_context           — current structured trip context
    context_enrichment_status — "not_started" | "completed" | "failed"
    planner_status         — "ready" | "partial_ready" | "missing_required_info"
    planner_task_results   — results returned by planner tool tasks
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
    travel_preferences: str

    is_admin: bool
    conversation_summary: str

    cache_status: str
    cache_similarity_score: float
    cache_matched_query: str
    cache_answer: str

    trip_context: dict
    context_enrichment_status: str
    planner_status: str
    planner_task_results: dict