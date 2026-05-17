from langchain_core.messages import HumanMessage
from langgraph.graph import END

from src.graph.state import AgentState
from src.utils.graph_guards import detect_repetition


def _current_turn_messages(messages: list) -> list:
    """
    Return only the messages belonging to the current turn.

    Slices from the last HumanMessage onwards so detect_repetition does not
    false-positive on tool calls from previous sessions.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


MAX_TOOL_CALLS = 8


def route_after_metadata(state: AgentState) -> str:
    """
    Conditional edge after extract_metadata.

    Strict architecture:
      Every user message must pass through validation before any intent routing.
    """
    return "validator"


def route_after_validator(state: AgentState) -> str:
    """
    Conditional edge after the validator node.

    If blocked, end immediately because the rejection message was already added
    to State by run_validator.

    If approved, continue to the master orchestrator.
    """
    status = state.get("validation_status", "approved")
    if status != "approved":
        return END

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
    If cache miss, continue to the legacy agent for now.

    Later:
      cache_miss → planner
    """
    if state.get("cache_status") == "hit":
        return END

    return "agent"


def should_continue(state: AgentState) -> str:
    """
    Conditional edge function — decides the next node after the legacy agent runs.

    Decision tree:
      1. tool_call_count >= MAX_TOOL_CALLS        → circuit_breaker
      2. Repetitive identical tool call detected  → circuit_breaker
      3. Last message contains tool_calls         → tools
      4. Full plan + admin session                → reviewer
      5. Final answer after cache miss            → cache_store
      6. Final answer otherwise                   → summarizer
    """
    last = state["messages"][-1]
    count = state.get("tool_call_count", 0)
    is_admin = state.get("is_admin", False)

    if count >= MAX_TOOL_CALLS:
        return "circuit_breaker"

    if detect_repetition(_current_turn_messages(state["messages"])):
        return "circuit_breaker"

    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"

    if is_admin and count >= 5 and state.get("current_city"):
        return "reviewer"

    if state.get("cache_status") == "miss":
        return "cache_store"

    return "summarizer"