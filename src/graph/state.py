from typing import Annotated, Dict, List, Optional, TypedDict
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

    planner_task_results       — raw results returned by planner tools/sub-agents
    planner_structured_results — typed shared planner outputs for sub-agents
    planner_dependency_graph   — explicit planner task dependency DAG
    planner_scheduler_result   — async scheduler waves and blocked tasks

    awaiting_user_clarification — True when planner stopped to ask a HITL question
    pending_trip_context        — saved partial TripContext waiting for user clarification
    pending_missing_fields      — required TripContext fields still missing
    pending_hitl_question       — last HITL question shown to the user
    pending_planner_task_results — planner tool results already collected before HITL stop
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
    planner_structured_results: dict
    planner_dependency_graph: dict
    planner_scheduler_result: dict

    awaiting_user_clarification: bool
    pending_trip_context: dict
    pending_missing_fields: List[str]
    pending_hitl_question: str
    pending_planner_task_results: dict