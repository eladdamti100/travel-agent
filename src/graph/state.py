"""
AgentState — central shared memory passed between all graph nodes.
"""

from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph.message import add_messages


# ── Typed sub-dicts ───────────────────────────────────────────────────────────

class BudgetBreakdownDict(TypedDict, total=False):
    budget: float
    flight_cost: float
    hotel_cost: float
    activities_cost: float
    total_estimated: float
    overage: float
    within_budget: bool


class CritiqueResultDict(TypedDict, total=False):
    passed: bool
    score: int
    reason: str
    issues: List[str]
    suggestions: List[str]
    completeness: Dict[str, Any]
    budget: BudgetBreakdownDict


# ── AgentState ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    Central shared memory across all graph nodes.

    Per-turn fields (reset by extract_metadata on each new user message):
        tool_call_count, force_replan, critic_attempts, hitl_decision,
        hitl_feedback, hitl_edit_attempts, orchestrator_route,
        orchestrator_reason, cache_status, cache_answer, cache_matched_query,
        cache_similarity_score, planner_status, planner_task_results,
        planner_structured_results, planner_dependency_graph,
        planner_scheduler_result, context_enrichment_status, used_web_source.

    Persistent fields (survive across turns):
        messages, preferred_airline, food_preference, num_travelers,
        travel_preferences, is_admin, conversation_summary, trip_context.

    HITL fields (set when planner stops for missing info):
        awaiting_user_clarification, pending_trip_context,
        pending_missing_fields, pending_hitl_question,
        pending_planner_task_results, planning_mode.
    """

    messages: Annotated[list, add_messages]

    # lightweight metadata
    current_city: str
    total_budget: float
    tool_call_count: int

    # validator & orchestrator
    validation_status: str
    orchestrator_route: str
    orchestrator_reason: str

    # persistent user preferences
    preferred_airline: str
    food_preference: str
    num_travelers: int
    travel_preferences: str

    # session flags
    is_admin: bool
    conversation_summary: str

    # semantic cache
    cache_status: str
    cache_similarity_score: float
    cache_matched_query: str
    cache_answer: str
    force_replan: bool
    planning_query: str  # raw user text captured by cache_checker before HITL

    # trip context — stored as dict so LangGraph can JSON-serialise it;
    # deserialise with: TripContext(**state["trip_context"])
    trip_context: Dict[str, Any]
    context_enrichment_status: str
    planner_status: str

    planner_task_results: Dict[str, str]         # task_type.value → raw result string
    planner_structured_results: Dict[str, Any]   # serialised PlannerToolResults
    planner_dependency_graph: Dict[str, Any]     # serialised PlannerDependencyGraph
    planner_scheduler_result: Dict[str, Any]     # serialised SchedulerResult

    # HITL clarification
    awaiting_user_clarification: bool
    pending_trip_context: Dict[str, Any]
    pending_missing_fields: List[str]
    pending_hitl_question: str
    pending_planner_task_results: Dict[str, str]
    planning_mode: str

    # critic & HITL approval
    critic_attempts: int
    hitl_decision: str
    critique_result: CritiqueResultDict
    hitl_feedback: str
    hitl_edit_attempts: int

    used_web_source: bool

    # Structured final plan — populated after a complete plan is generated.
    # Deserialise with: FinalPlan(**state["final_plan"])
    final_plan: Dict[str, Any]
