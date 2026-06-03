"""
Tests for the HITL graph — critic node, approval flow, routing, and replanning.

All tests are pure unit tests — no LLM calls, no real graph runs, no DB.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_plan_state(
    budget=3000.0,
    duration_days=7,
    num_travelers=1,
    flight_price=600.0,
    hotel_nightly=100.0,
    planner_status="ready",
    cache_status="miss",
    critic_attempts=0,
    hitl_decision="",
    hitl_feedback="",
    critique_passed=True,
    critique_score=8,
):
    flights = [{"price": flight_price}] if flight_price else []
    hotels = [{"price_per_night": hotel_nightly}] if hotel_nightly else []
    activities = [{"price": 30.0}, {"price": 50.0}]
    total = (flight_price or 0) + (hotel_nightly or 0) * duration_days + 80

    return {
        "total_budget": budget,
        "trip_context": {
            "destination_city": "Paris",
            "duration_days": duration_days,
            "num_travelers": num_travelers,
            "total_budget": budget,
        },
        "planner_structured_results": {
            "flights": flights,
            "hotels": hotels,
            "activities": activities,
            "visa": {"visa_required": False},
            "cost": {"total_cost": total},
        },
        "planner_task_results": {"fetch_flights": "some result"},
        "planner_status": planner_status,
        "cache_status": cache_status,
        "critic_attempts": critic_attempts,
        "hitl_decision": hitl_decision,
        "hitl_feedback": hitl_feedback,
        "critique_result": {
            "passed": critique_passed,
            "score": critique_score,
            "reason": "Test reason",
            "issues": [],
            "suggestions": [],
        },
        "messages": [HumanMessage(content="Plan a trip to Paris")],
        "awaiting_user_clarification": False,
        "force_replan": False,
    }


class TestStateFields:

    def test_critic_attempts_field_exists(self):
        from src.graph.state import AgentState
        assert "critic_attempts" in AgentState.__annotations__

    def test_hitl_decision_field_exists(self):
        from src.graph.state import AgentState
        assert "hitl_decision" in AgentState.__annotations__

    def test_critique_result_field_exists(self):
        from src.graph.state import AgentState
        assert "critique_result" in AgentState.__annotations__

    def test_hitl_feedback_field_exists(self):
        from src.graph.state import AgentState
        assert "hitl_feedback" in AgentState.__annotations__

    def test_critic_attempts_is_int_type(self):
        from src.graph.state import AgentState
        assert AgentState.__annotations__["critic_attempts"] == int

    def test_hitl_decision_is_str_type(self):
        from src.graph.state import AgentState
        assert AgentState.__annotations__["hitl_decision"] == str

    def test_critique_result_is_dict_type(self):
        from src.graph.state import AgentState
        assert AgentState.__annotations__["critique_result"] == dict

    def test_hitl_feedback_is_str_type(self):
        from src.graph.state import AgentState
        assert AgentState.__annotations__["hitl_feedback"] == str


class TestCriticNode:

    def test_critic_node_returns_critic_attempts(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(budget=3000, flight_price=600, hotel_nightly=100)
        result = critic_node(state)
        assert "critic_attempts" in result
        assert result["critic_attempts"] == 1

    def test_critic_node_increments_attempts(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(critic_attempts=1)
        result = critic_node(state)
        assert result["critic_attempts"] == 2

    def test_critic_node_returns_critique_result(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(budget=3000)
        result = critic_node(state)
        assert "critique_result" in result
        cr = result["critique_result"]
        assert "passed" in cr
        assert "score" in cr
        assert "reason" in cr
        assert "issues" in cr
        assert "suggestions" in cr
        assert "completeness" in cr
        assert "budget" in cr

    def test_critic_node_passes_within_budget(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(budget=5000, flight_price=600, hotel_nightly=100)
        result = critic_node(state)
        assert result["critique_result"]["passed"] is True

    def test_critic_node_fails_over_budget(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(budget=500, flight_price=800, hotel_nightly=150)
        result = critic_node(state)
        assert result["critique_result"]["passed"] is False

    def test_critic_node_score_is_between_0_and_10(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(budget=500, flight_price=800, hotel_nightly=150)
        result = critic_node(state)
        assert 0 <= result["critique_result"]["score"] <= 10

    def test_critic_node_suggestions_are_specific(self):
        from src.graph.nodes import critic_node
        state = _make_plan_state(budget=500, flight_price=800, hotel_nightly=150)
        result = critic_node(state)
        suggestions = result["critique_result"]["suggestions"]
        assert len(suggestions) > 0
        for s in suggestions:
            assert "$" in s or "day" in s.lower()

    def test_max_critic_attempts_is_2(self):
        from src.graph.nodes import MAX_CRITIC_ATTEMPTS
        assert MAX_CRITIC_ATTEMPTS == 2


class TestHitlApprovalNode:

    def test_hitl_approval_node_exists(self):
        from src.graph.nodes import hitl_approval_node
        assert callable(hitl_approval_node)

    def test_hitl_approval_node_uses_interrupt(self):
        import inspect
        from src.graph.nodes import hitl_approval_node
        source = inspect.getsource(hitl_approval_node)
        assert "interrupt" in source

    def test_hitl_approval_node_reads_critique_result(self):
        import inspect
        from src.graph.nodes import hitl_approval_node
        source = inspect.getsource(hitl_approval_node)
        assert "critique_result" in source

    def test_hitl_approval_sets_force_replan_on_edit(self):
        import inspect
        from src.graph.nodes import hitl_approval_node
        source = inspect.getsource(hitl_approval_node)
        assert "force_replan" in source
        assert "edit" in source

    def test_hitl_approval_returns_hitl_decision(self):
        import inspect
        from src.graph.nodes import hitl_approval_node
        source = inspect.getsource(hitl_approval_node)
        assert "hitl_decision" in source

    def test_hitl_approval_returns_hitl_feedback(self):
        import inspect
        from src.graph.nodes import hitl_approval_node
        source = inspect.getsource(hitl_approval_node)
        assert "hitl_feedback" in source


class TestGraphTopology:

    def test_critic_node_registered_in_graph(self):
        from src.graph.workflow import graph
        assert "critic" in graph.nodes

    def test_hitl_approval_node_registered_in_graph(self):
        from src.graph.workflow import graph
        assert "hitl_approval" in graph.nodes

    def test_master_planner_registered_in_graph(self):
        from src.graph.workflow import graph
        assert "master_planner" in graph.nodes

    def test_cache_store_registered_in_graph(self):
        from src.graph.workflow import graph
        assert "cache_store" in graph.nodes

    def test_all_required_nodes_present(self):
        from src.graph.workflow import graph
        required = {
            "extract_metadata", "validator", "master_orchestrator",
            "master_planner", "critic", "hitl_approval", "cache_store",
            "summarizer"
        }
        for node in required:
            assert node in graph.nodes, f"Missing node: {node}"


class TestRoutingLogic:
    def test_missing_info_routes_to_end(self):
        from src.graph.router import route_after_master_planner
        state = {"planner_status": "missing_required_info"}
        assert route_after_master_planner(state) == END

    def test_ready_status_routes_to_critic(self):
        from src.graph.router import route_after_master_planner
        state = {"planner_status": "ready", "cache_status": "miss"}
        assert route_after_master_planner(state) == "critic"

    def test_any_complete_plan_routes_to_critic(self):
        from src.graph.router import route_after_master_planner
        assert route_after_master_planner({}) == "critic"
        assert route_after_master_planner({"cache_status": "hit"}) == "critic"
        assert route_after_master_planner({"cache_status": "miss"}) == "critic"


    def test_critic_always_routes_to_hitl(self):
        from src.graph.router import route_after_critic
        assert route_after_critic({"critique_result": {"passed": True}, "critic_attempts": 1}) == "hitl_approval"
        assert route_after_critic({"critique_result": {"passed": False}, "critic_attempts": 1}) == "hitl_approval"
        assert route_after_critic({"critique_result": {"passed": False}, "critic_attempts": 2}) == "hitl_approval"
        assert route_after_critic({}) == "hitl_approval"


    def test_approved_routes_to_cache_store(self):
        from src.graph.router import route_after_hitl
        assert route_after_hitl({"hitl_decision": "approved"}) == "cache_store"

    def test_edit_routes_to_master_planner(self):
        from src.graph.router import route_after_hitl
        assert route_after_hitl({"hitl_decision": "edit"}) == "master_planner"

    def test_cancelled_routes_to_end(self):
        from src.graph.router import route_after_hitl
        assert route_after_hitl({"hitl_decision": "cancelled"}) == END

    def test_default_empty_decision_routes_to_cache_store(self):
        from src.graph.router import route_after_hitl
        assert route_after_hitl({}) == "cache_store"
        assert route_after_hitl({"hitl_decision": ""}) == "cache_store"


class TestTerminalFlow:

    def test_prompt_plan_approval_exists(self):
        from src.main import _prompt_plan_approval
        assert callable(_prompt_plan_approval)

    def test_prompt_returns_approved_on_a(self, monkeypatch):
        from src.main import _prompt_plan_approval
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: "a")
        result = _prompt_plan_approval({"score": 8, "reason": "ok", "issues": [], "suggestions": []})
        assert result["decision"] == "approved"
        assert result["feedback"] == ""

    def test_prompt_returns_cancelled_on_c(self, monkeypatch):
        from src.main import _prompt_plan_approval
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: "c")
        result = _prompt_plan_approval({"score": 5, "reason": "bad", "issues": [], "suggestions": []})
        assert result["decision"] == "cancelled"

    def test_prompt_returns_edit_with_feedback(self, monkeypatch):
        from src.main import _prompt_plan_approval
        responses = iter(["e", "find cheaper hotels"])
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: next(responses))
        result = _prompt_plan_approval({"score": 3, "reason": "bad", "issues": [], "suggestions": []})
        assert result["decision"] == "edit"
        assert result["feedback"] == "find cheaper hotels"

    def test_prompt_accepts_full_word_approve(self, monkeypatch):
        from src.main import _prompt_plan_approval
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: "approve")
        result = _prompt_plan_approval({"score": 9, "reason": "good", "issues": [], "suggestions": []})
        assert result["decision"] == "approved"

    def test_prompt_accepts_full_word_cancel(self, monkeypatch):
        from src.main import _prompt_plan_approval
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: "cancel")
        result = _prompt_plan_approval({"score": 9, "reason": "good", "issues": [], "suggestions": []})
        assert result["decision"] == "cancelled"

    def test_prompt_retries_on_invalid_input(self, monkeypatch):
        from src.main import _prompt_plan_approval
        responses = iter(["x", "y", "a"])
        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: next(responses))
        result = _prompt_plan_approval({"score": 8, "reason": "ok", "issues": [], "suggestions": []})
        assert result["decision"] == "approved"

    def test_stream_graph_helper_exists(self):
        import inspect
        from src.main import run
        source = inspect.getsource(run)
        assert "_stream_graph" in source
        assert "pending_interrupt" in source
        assert "Command" in source


class TestReplanningOnEdit:

    def test_hitl_feedback_field_in_replanning_result(self):
        from src.agents.sub_agents.replanning_agent import ReplanningResult
        import inspect
        fields = [f.name for f in ReplanningResult.__dataclass_fields__.values()]
        assert "hitl_feedback" in fields

    def test_analyze_replanning_accepts_hitl_feedback(self):
        from src.agents.sub_agents.replanning_agent import analyze_replanning
        import inspect
        sig = inspect.signature(analyze_replanning)
        assert "hitl_feedback" in sig.parameters

    def test_analyze_replanning_carries_feedback_in_result(self):
        from src.agents.sub_agents.replanning_agent import analyze_replanning
        from src.models.trip_context import TripContext

        ctx = TripContext(
            destination_city="Paris",
            duration_days=7,
            total_budget=500.0,
        )
        result = analyze_replanning(
            old_context=ctx,
            new_context=ctx,
            existing_results={},
            hitl_feedback="find cheaper hotels",
        )
        assert result.hitl_feedback == "find cheaper hotels"

    def test_hitl_feedback_flows_to_planner(self):
        import inspect
        from src.agents.planner import _run_master_planner_async
        source = inspect.getsource(_run_master_planner_async)
        assert "hitl_feedback" in source

    def test_hitl_feedback_flows_to_notes_section(self):
        import inspect
        from src.agents.planner import _generate_notes_section
        source = inspect.getsource(_generate_notes_section)
        assert "hitl_feedback" in source
        assert "User requested changes" in source

    def test_force_replan_set_on_edit_in_hitl_node(self):
        import inspect
        from src.graph.nodes import hitl_approval_node
        source = inspect.getsource(hitl_approval_node)
        assert "force_replan" in source
        assert 'decision == "edit"' in source or "decision == 'edit'" in source

    def test_planner_clears_hitl_feedback_after_completion(self):
        import inspect
        from src.agents.planner import _run_master_planner_async
        source = inspect.getsource(_run_master_planner_async)
        assert "hitl_feedback" in source
        assert '""' in source or "= 0" in source


class TestDemoScenarios:

    def test_demo_scenarios_file_exists(self):
        from pathlib import Path
        demo_file = Path("docs/demo_scenarios.md")
        assert demo_file.exists(), "docs/demo_scenarios.md not found"

    def test_demo_file_has_paris_scenario(self):
        from pathlib import Path
        content = Path("docs/demo_scenarios.md").read_text(encoding="utf-8")
        assert "Paris" in content
        assert "$500" in content

    def test_demo_file_has_normal_trip_scenario(self):
        from pathlib import Path
        content = Path("docs/demo_scenarios.md").read_text(encoding="utf-8")
        assert "Demo 2" in content
        assert "Approve" in content
        assert "Edit" in content
        assert "Cancel" in content

    def test_demo_file_has_verification_checklist(self):
        from pathlib import Path
        content = Path("docs/demo_scenarios.md").read_text(encoding="utf-8")
        assert "checklist" in content.lower() or "Verification" in content

    def test_paris_500_critic_detects_over_budget(self):
        from src.agents.critic import critique_plan
        state = {
            "total_budget": 500.0,
            "trip_context": {"duration_days": 7, "num_travelers": 1, "total_budget": 500.0},
            "planner_structured_results": {
                "flights": [],
                "hotels": [{"price_per_night": 85.0}],
                "activities": [{"price": 20.0}, {"price": 35.0}, {"price": 95.0}],
                "visa": None,
                "cost": {"total_cost": 745.0},
            },
            "planner_task_results": {},
        }
        result = critique_plan(state)
        assert result.passed is False
        assert result.budget.total_estimated == 745.0
        assert result.budget.overage == 245.0
        assert len(result.suggestions) > 0

    def test_normal_trip_critic_passes(self):
        from src.agents.critic import critique_plan
        state = {
            "total_budget": 3000.0,
            "trip_context": {"duration_days": 7, "num_travelers": 2, "total_budget": 3000.0},
            "planner_structured_results": {
                "flights": [{"price": 800.0}],
                "hotels": [{"price_per_night": 100.0}],
                "activities": [{"price": 50.0}, {"price": 30.0}],
                "visa": {"visa_required": False},
                "cost": {"total_cost": 2480.0},
            },
            "planner_task_results": {},
        }
        result = critique_plan(state)
        assert result.passed is True
        assert result.budget.within_budget is True


class TestFullRoutingChain:

    def test_full_approve_chain(self):
        from src.graph.router import (
            route_after_master_planner,
            route_after_critic,
            route_after_hitl,
        )
        state = {"planner_status": "ready", "cache_status": "miss"}

        step1 = route_after_master_planner(state)
        assert step1 == "critic"

        state["critique_result"] = {"passed": True}
        step2 = route_after_critic(state)
        assert step2 == "hitl_approval"

        state["hitl_decision"] = "approved"
        step3 = route_after_hitl(state)
        assert step3 == "cache_store"

    def test_full_cancel_chain(self):
        from src.graph.router import (
            route_after_master_planner,
            route_after_critic,
            route_after_hitl,
        )
        state = {"planner_status": "ready"}
        assert route_after_master_planner(state) == "critic"
        assert route_after_critic(state) == "hitl_approval"
        state["hitl_decision"] = "cancelled"
        assert route_after_hitl(state) == END

    def test_full_edit_chain(self):
        from src.graph.router import (
            route_after_master_planner,
            route_after_critic,
            route_after_hitl,
        )
        state = {"planner_status": "ready"}
        assert route_after_master_planner(state) == "critic"
        assert route_after_critic(state) == "hitl_approval"
        state["hitl_decision"] = "edit"
        assert route_after_hitl(state) == "master_planner"

    def test_missing_info_never_reaches_critic(self):
        from src.graph.router import route_after_master_planner
        state = {"planner_status": "missing_required_info"}
        result = route_after_master_planner(state)
        assert result == END
        assert result != "critic"

    def test_extract_metadata_resets_critic_attempts(self):
        from src.graph.nodes import extract_metadata
        from langchain_core.messages import HumanMessage
        state = {
            "messages": [HumanMessage(content="Plan a trip")],
            "critic_attempts": 5,
            "hitl_decision": "edit",
            "hitl_feedback": "old feedback",
        }
        result = extract_metadata(state)
        assert result["critic_attempts"] == 0
        assert result["hitl_decision"] == ""
        assert result["hitl_feedback"] == ""
