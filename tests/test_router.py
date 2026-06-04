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
        # New HITL topology: a complete plan routes to the critic (Session 7),
        # which then drives the critic → hitl_approval → cache_store chain.
        from src.graph.router import route_after_master_planner
        assert route_after_master_planner({"planner_status": "missing_required_info"}) == END
        assert route_after_master_planner({"cache_status": "miss"}) == "critic"
        assert route_after_master_planner({}) == "critic"

    def test_legacy_should_continue_removed(self):
        """should_continue was deleted in P2-1.3 (dead code removal)."""
        import src.graph.router as router_mod
        assert not hasattr(router_mod, "should_continue"), (
            "should_continue was the legacy agent/tools loop router — it should be gone"
        )
