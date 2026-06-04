"""
Conditional edge functions for the LangGraph travel planner workflow.
"""

from langgraph.graph import END

from src.config.settings import settings
from src.graph.state import AgentState

# Imported here to avoid a circular import (nodes.py → router.py is one-way).
from src.graph.nodes import MAX_CRITIC_ATTEMPTS

MAX_HITL_EDIT_ATTEMPTS: int = settings.max_hitl_edit_attempts


def route_after_validator(state: AgentState) -> str:
    """
    Conditional edge after the validator node.

    If blocked, end immediately because the rejection message was already added
    to State by run_validator.

    If approved and the previous planner turn is waiting for HITL clarification,
    bypass the master orchestrator and resume the existing planning flow.

    Otherwise, continue to the master orchestrator.
    """
    status = state.get("validation_status", "approved")
    if status != "approved":
        return END

    if state.get("awaiting_user_clarification"):
        return "resume_hitl_context"

    return "master_orchestrator"


def route_after_orchestrator(state: AgentState) -> str:
    """
    Conditional edge after the master orchestrator.

    The orchestrator writes orchestrator_route into State using one of:
      - preferences_memory
      - research
      - cache_check

    Unknown routes are treated as cache_check because that is the safest
    default for full trip-planning requests.
    """
    route = state.get("orchestrator_route", "cache_check")

    if route == "preferences_memory":
        return "preferences_memory"

    if route == "research":
        return "researcher"

    return "cache_check"


def route_after_cache_check(state: AgentState) -> str:
    """
    Conditional edge after semantic cache check.

    If cache hit, the cached answer was already added to messages, so the graph ends.
    If cache miss, continue to the master planner.
    """
    if state.get("cache_status") == "hit":
        return END

    return "master_planner"


def route_after_master_planner(state: AgentState) -> str:
    """
    Conditional edge after the new master planner.

    HITL stop (missing trip details) ends the graph immediately — the question
    was already added to messages by the planner.

    A complete plan always proceeds to the critic before any caching.
    """
    if state.get("planner_status") == "missing_required_info":
        return END

    return "critic"


def route_after_critic(state: AgentState) -> str:
    """
    Conditional edge after the critic node.

    If the critic failed AND we haven't reached the attempt cap yet,
    route back to master_planner so the graph auto-replans with the
    critic's suggestions injected into the prompt.

    Once the plan passes OR the attempt cap is reached, hand control
    to the human via hitl_approval so they can approve, edit, or cancel.
    """
    critique = state.get("critique_result") or {}
    attempts = state.get("critic_attempts") or 0

    if not critique.get("passed", True) and attempts < MAX_CRITIC_ATTEMPTS:
        return "master_planner"

    return "hitl_approval"


def route_after_hitl(state: AgentState) -> str:
    """
    Conditional edge after the hitl_approval node.

    approved  → cache_store (then summarizer → END)
    edit      → master_planner (hitl_feedback is in state for the replanner)
              → END when MAX_HITL_EDIT_ATTEMPTS is exceeded (prevents infinite loop)
    cancelled → END
    """
    decision = state.get("hitl_decision", "approved")

    if decision == "cancelled":
        return END

    if decision == "edit":
        edit_attempts = state.get("hitl_edit_attempts") or 0
        if edit_attempts > MAX_HITL_EDIT_ATTEMPTS:
            # Inject a graceful message — state["messages"] is append-only so
            # we cannot write here; the guard just ends the graph.  The HITL
            # approval node already logged the attempt count.
            return END
        return "master_planner"

    return "cache_store"