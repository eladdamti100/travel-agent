"""
Tests for graph routing functions — router.py
All routing functions are pure state → str. No mocking needed.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END


class TestRouterFunctions:

    def test_validator_blocked_ends_graph(self):
        from src.graph.router import route_after_validator
        for verdict in ("blocked_harm", "blocked_injection", "blocked_scope", "blocked_city"):
            state = {"validation_status": verdict, "awaiting_user_clarification": False}
            assert route_after_validator(state) == END, f"Expected END for {verdict}"

    def test_validator_approved_routes_correctly(self):
        from src.graph.router import route_after_validator
        assert route_after_validator({"validation_status": "approved"}) == "master_orchestrator"
        assert route_after_validator({
            "validation_status": "approved",
            "awaiting_user_clarification": True,
        }) == "resume_hitl_context"

    def test_orchestrator_all_branches(self):
        from src.graph.router import route_after_orchestrator
        assert route_after_orchestrator({"orchestrator_route": "preferences_memory"}) == "preferences_memory"
        assert route_after_orchestrator({"orchestrator_route": "research"}) == "researcher"
        assert route_after_orchestrator({"orchestrator_route": "cache_check"}) == "cache_check"
        assert route_after_orchestrator({}) == "cache_check"  # unknown route → safe default

    def test_cache_check_routing(self):
        from src.graph.router import route_after_cache_check
        assert route_after_cache_check({"cache_status": "hit"}) == END
        assert route_after_cache_check({"cache_status": "miss"}) == "master_planner"

    def test_master_planner_routing(self):
        from src.graph.router import route_after_master_planner
        assert route_after_master_planner({"planner_status": "missing_required_info"}) == END
        assert route_after_master_planner({"cache_status": "miss"}) == "cache_store"
        assert route_after_master_planner({}) == "summarizer"

    def test_should_continue_all_branches(self):
        from src.graph.router import should_continue, MAX_TOOL_CALLS

        base = [HumanMessage(content="Plan a Paris trip")]

        # Max tool calls → circuit_breaker
        assert should_continue({
            "messages": base + [AIMessage(content="done")],
            "tool_call_count": MAX_TOOL_CALLS,
        }) == "circuit_breaker"

        # Tool calls in last message → tools
        tool_msg = AIMessage(content="", tool_calls=[{"name": "fetch_flights", "args": {}, "id": "1"}])
        assert should_continue({
            "messages": base + [tool_msg],
            "tool_call_count": 0,
        }) == "tools"

        # Admin session with enough calls → reviewer
        assert should_continue({
            "messages": base + [AIMessage(content="Final plan")],
            "tool_call_count": 6,
            "is_admin": True,
            "current_city": "Paris",
        }) == "reviewer"

        # Cache miss final answer → cache_store
        assert should_continue({
            "messages": base + [AIMessage(content="Final plan")],
            "tool_call_count": 0,
            "cache_status": "miss",
        }) == "cache_store"

        # Default → summarizer
        assert should_continue({
            "messages": base + [AIMessage(content="Final plan")],
            "tool_call_count": 0,
        }) == "summarizer"
