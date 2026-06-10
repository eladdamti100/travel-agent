"""
Regression tests for the four P0 (blocking for v2) fixes.

P0-2.2  run_master_planner no longer calls asyncio.run() directly —
        it uses a ThreadPoolExecutor so it is safe in already-running loops.

P0-2.4  web_research_tavily uses asyncio.to_thread for the blocking
        Tavily .invoke() call instead of calling it directly in async context.

P0-3.1  HITL edit loop is capped at MAX_HITL_EDIT_ATTEMPTS.
        After the cap the node overrides decision→"cancelled" with a message,
        and route_after_hitl routes to END.

P0-4.2  extract_metadata resets all per-turn output fields so stale
        data from a previous plan cannot bleed into the current turn.
"""

import asyncio
import inspect
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END


# ─────────────────────────────────────────────────────────────────────────────
# P0-2.2  asyncio.run() replaced with ThreadPoolExecutor
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncioRunFix:
    """run_master_planner must not call asyncio.run() directly."""

    def test_uses_thread_pool_executor(self):
        import ast
        from src.agents.planner import run_master_planner
        src = inspect.getsource(run_master_planner)
        assert "ThreadPoolExecutor" in src, "run_master_planner must use ThreadPoolExecutor"

        # Parse the source and collect every Call node's function name.
        # This correctly ignores asyncio.run() references inside docstrings.
        tree = ast.parse(inspect.getsource(inspect.getmodule(run_master_planner)))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "run_master_planner":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            calls.append(f"{getattr(child.func.value, 'id', '?')}.{child.func.attr}")
                        elif isinstance(child.func, ast.Name):
                            calls.append(child.func.id)
                break

        assert "asyncio.run" not in calls, (
            f"asyncio.run() must not be called inside run_master_planner, found calls: {calls}"
        )

    def test_concurrent_futures_imported(self):
        import src.agents.planner as planner_mod
        import concurrent.futures  # noqa: F401 — ensure the stdlib module itself exists
        src = inspect.getsource(planner_mod)
        assert "concurrent.futures" in src, (
            "concurrent.futures must be imported at module level in planner.py"
        )

    def test_planner_callable_from_inside_running_loop(self):
        """
        Simulate the crash scenario: calling run_master_planner from an already-
        running event loop (FastAPI / LangGraph server / pytest-asyncio).
        With asyncio.run() this raises RuntimeError; with ThreadPoolExecutor it works.
        """
        from src.agents.planner import run_master_planner

        dummy_result = {"planner_status": "ready", "messages": [AIMessage(content="ok")]}

        async def _inner():
            with patch(
                "src.agents.planner._run_master_planner_async",
                new=AsyncMock(return_value=dummy_result),
            ):
                # If asyncio.run() were used this would raise
                # "This event loop is already running".
                result = run_master_planner({"messages": []})
                return result

        result = asyncio.run(_inner())
        assert result["planner_status"] == "ready"


# ─────────────────────────────────────────────────────────────────────────────
# P0-2.4  web_research_tavily uses asyncio.to_thread
# ─────────────────────────────────────────────────────────────────────────────

class TestTavilyAsyncFix:
    """web_research_tavily must not block the event loop."""

    def test_uses_asyncio_to_thread(self):
        from src.tools.web_api_tools import _tavily_search
        src = inspect.getsource(_tavily_search)
        assert "asyncio.to_thread" in src, (
            "web_research_tavily must call asyncio.to_thread to avoid blocking the event loop"
        )

    def test_asyncio_imported_in_web_api_tools(self):
        import src.tools.web_api_tools as mod
        src = inspect.getsource(mod)
        assert "import asyncio" in src, "asyncio must be imported at module level in web_api_tools.py"

    def test_tavily_invoke_not_called_directly(self):
        """search.invoke() must not appear as a bare synchronous call in the async body."""
        from src.tools.web_api_tools import web_research_tavily
        src = inspect.getsource(web_research_tavily.coroutine)
        for line in src.splitlines():
            stripped = line.strip()
            # Skip comment lines — they may mention the old pattern for documentation.
            if stripped.startswith("#"):
                continue
            # Any non-comment line that contains search.invoke must route through to_thread.
            if "search.invoke" in stripped and "asyncio.to_thread" not in stripped:
                pytest.fail(
                    f"search.invoke() called without asyncio.to_thread: {stripped!r}"
                )

    @pytest.mark.asyncio
    async def test_tavily_missing_key_returns_unavailable_message(self):
        from src.tools.web_api_tools import web_research_tavily
        with patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False):
            result = await web_research_tavily.ainvoke({"query": "Paris travel tips"})
        # The exact wording changed in P2-1.5 (log standardisation); check for
        # any keyword indicating the key is absent and search is unavailable.
        assert any(kw in result.lower() for kw in ("unavailable", "not configured", "missing", "dormant"))

    @pytest.mark.asyncio
    async def test_tavily_does_not_block_event_loop_on_error(self):
        """Even when Tavily raises, the tool must return a string (not propagate)."""
        from src.tools.web_api_tools import web_research_tavily
        with patch.dict("os.environ", {"TAVILY_API_KEY": "fake-key"}):
            with patch("asyncio.to_thread", new=AsyncMock(side_effect=RuntimeError("network down"))):
                result = await web_research_tavily.ainvoke({"query": "Paris"})
        assert isinstance(result, str)
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# P0-3.1  HITL edit loop cap
# ─────────────────────────────────────────────────────────────────────────────

class TestHitlEditCap:
    """After MAX_HITL_EDIT_ATTEMPTS the node must end the session gracefully."""

    def test_max_hitl_edit_attempts_constant_exists(self):
        from src.graph.router import MAX_HITL_EDIT_ATTEMPTS
        assert isinstance(MAX_HITL_EDIT_ATTEMPTS, int)
        assert MAX_HITL_EDIT_ATTEMPTS >= 2, "cap must allow at least 2 edits"

    def test_hitl_edit_attempts_field_in_state(self):
        from src.graph.state import AgentState
        assert "hitl_edit_attempts" in AgentState.__annotations__
        assert AgentState.__annotations__["hitl_edit_attempts"] == int

    def test_edit_under_cap_routes_to_master_planner(self):
        from src.graph.router import route_after_hitl, MAX_HITL_EDIT_ATTEMPTS
        state = {
            "hitl_decision": "edit",
            "hitl_edit_attempts": MAX_HITL_EDIT_ATTEMPTS - 1,
        }
        assert route_after_hitl(state) == "master_planner"

    def test_edit_at_cap_routes_to_end(self):
        from src.graph.router import route_after_hitl, MAX_HITL_EDIT_ATTEMPTS
        state = {
            "hitl_decision": "edit",
            "hitl_edit_attempts": MAX_HITL_EDIT_ATTEMPTS + 1,
        }
        assert route_after_hitl(state) == END

    def test_approved_always_routes_to_cache_store_regardless_of_edit_attempts(self):
        from src.graph.router import route_after_hitl, MAX_HITL_EDIT_ATTEMPTS
        state = {
            "hitl_decision": "approved",
            "hitl_edit_attempts": MAX_HITL_EDIT_ATTEMPTS + 10,
        }
        assert route_after_hitl(state) == "cache_store"

    def test_cancelled_always_routes_to_end_regardless_of_edit_attempts(self):
        from src.graph.router import route_after_hitl, MAX_HITL_EDIT_ATTEMPTS
        state = {
            "hitl_decision": "cancelled",
            "hitl_edit_attempts": 0,
        }
        assert route_after_hitl(state) == END

    def test_hitl_node_increments_edit_attempts_on_edit(self):
        """hitl_approval_node must increment hitl_edit_attempts when decision=edit."""
        src = inspect.getsource(
            __import__("src.graph.nodes", fromlist=["hitl_approval_node"]).hitl_approval_node
        )
        assert "hitl_edit_attempts" in src
        assert "edit_attempts" in src

    def test_hitl_node_injects_message_at_cap(self):
        """When cap is exceeded the node must return a user-visible AIMessage."""
        src = inspect.getsource(
            __import__("src.graph.nodes", fromlist=["hitl_approval_node"]).hitl_approval_node
        )
        assert "MAX_HITL_EDIT_ATTEMPTS" in src
        assert "AIMessage" in src

    def test_extract_metadata_resets_edit_attempts(self):
        from src.graph.nodes import extract_metadata
        state = {
            "messages": [HumanMessage(content="New trip to Tokyo")],
            "hitl_edit_attempts": 5,
        }
        result = extract_metadata(state)
        assert result.get("hitl_edit_attempts") == 0, (
            "extract_metadata must reset hitl_edit_attempts to 0 on each new turn"
        )


# ─────────────────────────────────────────────────────────────────────────────
# P0-4.2  Stale state reset in extract_metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestStaleStateReset:
    """extract_metadata must zero out all per-turn output fields."""

    # Fields that must be reset to a falsy value on every new turn.
    _TURN_FIELDS = {
        "tool_call_count": 0,
        "force_replan": False,
        "critic_attempts": 0,
        "hitl_decision": "",
        "hitl_feedback": "",
        "hitl_edit_attempts": 0,
        "orchestrator_route": "",
        "orchestrator_reason": "",
        "cache_status": "",
        "cache_answer": "",
        "cache_matched_query": "",
        "cache_similarity_score": 0.0,
        "planner_status": "",
        "planner_task_results": {},
        "planner_structured_results": {},
        "planner_dependency_graph": {},
        "planner_scheduler_result": {},
        "context_enrichment_status": "",
        "used_web_source": False,
    }

    def _run(self, stale_state: dict) -> dict:
        from src.graph.nodes import extract_metadata
        stale_state.setdefault("messages", [HumanMessage(content="New query")])
        return extract_metadata(stale_state)

    @pytest.mark.parametrize("field,expected_reset", list(_TURN_FIELDS.items()))
    def test_field_is_reset(self, field, expected_reset):
        stale = {field: "STALE_VALUE_123"}
        result = self._run(stale)
        assert field in result, f"extract_metadata must reset '{field}'"
        assert result[field] == expected_reset, (
            f"'{field}' expected {expected_reset!r}, got {result[field]!r}"
        )

    def test_stale_planner_results_cleared(self):
        stale = {
            "planner_task_results": {"fetch_flights": "old flight data"},
            "planner_structured_results": {"flights": [{"price": 999}]},
        }
        result = self._run(stale)
        assert result["planner_task_results"] == {}
        assert result["planner_structured_results"] == {}

    def test_stale_cache_answer_cleared(self):
        stale = {"cache_answer": "This is a cached plan from last turn"}
        result = self._run(stale)
        assert result["cache_answer"] == ""

    def test_stale_orchestrator_route_cleared(self):
        stale = {"orchestrator_route": "research", "orchestrator_reason": "old reason"}
        result = self._run(stale)
        assert result["orchestrator_route"] == ""
        assert result["orchestrator_reason"] == ""

    def test_persistent_fields_not_cleared(self):
        """Preferences and session state must survive across turns."""
        from src.graph.nodes import extract_metadata
        state = {
            "messages": [HumanMessage(content="New trip")],
            "preferred_airline": "El Al",
            "food_preference": "kosher",
            "num_travelers": 2,
            "travel_preferences": "- Prefers window seat",
            "is_admin": True,
            "conversation_summary": "Previous: Paris trip was planned.",
        }
        result = extract_metadata(state)
        # These keys must NOT be reset (they persist across the session)
        for field in ("preferred_airline", "food_preference", "num_travelers",
                      "travel_preferences", "is_admin", "conversation_summary"):
            assert field not in result, (
                f"extract_metadata must not reset persistent field '{field}'"
            )
