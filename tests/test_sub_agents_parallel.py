"""
Tests for parallel sub-agent execution — run_sub_agents_async (delegates to WebSupervisor).
"""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


def _make_trip_context(**overrides):
    from src.models.trip_context import TripContext
    defaults = dict(
        origin_airport="TLV", origin_country="Israel",
        destination_city="Paris", destination_country="France",
        duration_days=5, total_budget=2000.0,
    )
    defaults.update(overrides)
    return TripContext(**defaults)


def _mock_agent(result_keys, raw_results=None, crash=False):
    m = MagicMock()
    m.result_keys = result_keys
    m.agent_name = result_keys[0]
    if crash:
        async def _fail(**kw): raise RuntimeError("agent down")
        m.run = _fail
    else:
        from src.models.planner import PlannerToolResults
        r = MagicMock(spec=PlannerToolResults)
        r.raw_results = raw_results or {}
        m.run = AsyncMock(return_value=r)
    return m


class TestSubAgentsParallel:

    def test_all_agents_run_and_results_merged(self):
        from src.agents.planner import run_sub_agents_async
        transport = _mock_agent(("fetch_flights", "check_visa"), {"fetch_flights": '{"airline": "El Al"}'})
        stay = _mock_agent(("fetch_hotels",), {"fetch_hotels": '{"name": "Ibis"}'})
        experience = _mock_agent(("fetch_activities",), {"fetch_activities": '{"activity": "Louvre"}'})
        with patch("src.agents.web_supervisor.get_planner_agents", return_value=[transport, stay, experience]):
            merged = asyncio.run(run_sub_agents_async(context=_make_trip_context()))
        assert "fetch_flights" in merged and "fetch_hotels" in merged and "fetch_activities" in merged

    def test_failed_agent_does_not_block_others(self):
        from src.agents.planner import run_sub_agents_async
        transport = _mock_agent(("fetch_flights", "check_visa"), crash=True)
        stay = _mock_agent(("fetch_hotels",), {"fetch_hotels": '{"name": "Ibis"}'})
        experience = _mock_agent(("fetch_activities",), {"fetch_activities": '{"activity": "Louvre"}'})
        with patch("src.agents.web_supervisor.get_planner_agents", return_value=[transport, stay, experience]):
            merged = asyncio.run(run_sub_agents_async(context=_make_trip_context()))
        assert "fetch_flights" not in merged
        assert "fetch_hotels" in merged and "fetch_activities" in merged

    def test_skip_optimization_and_result_preservation(self):
        """When transport's keys are already covered its run is skipped, existing results preserved."""
        from src.agents.planner import run_sub_agents_async
        existing = {"fetch_flights": '{"airline": "El Al"}', "check_visa": "No visa required."}
        transport = _mock_agent(("fetch_flights", "check_visa"))
        stay = _mock_agent(("fetch_hotels",), {"fetch_hotels": '{"name": "Ibis"}'})
        experience = _mock_agent(("fetch_activities",), {"fetch_activities": '{"activity": "Louvre"}'})
        with patch("src.agents.web_supervisor.get_planner_agents", return_value=[transport, stay, experience]):
            merged = asyncio.run(run_sub_agents_async(context=_make_trip_context(), existing_results=existing))
        transport.run.assert_not_awaited()
        assert "El Al" in merged["fetch_flights"]
        assert "fetch_hotels" in merged

    def test_all_agents_skipped_when_fully_covered(self):
        from src.agents.planner import run_sub_agents_async
        existing = {"fetch_flights": "...", "check_visa": "...", "fetch_hotels": "...", "fetch_activities": "..."}
        transport = _mock_agent(("fetch_flights", "check_visa"))
        stay = _mock_agent(("fetch_hotels",))
        experience = _mock_agent(("fetch_activities",))
        with patch("src.agents.web_supervisor.get_planner_agents", return_value=[transport, stay, experience]):
            merged = asyncio.run(run_sub_agents_async(context=_make_trip_context(), existing_results=existing))
        transport.run.assert_not_awaited()
        stay.run.assert_not_awaited()
        experience.run.assert_not_awaited()
        assert merged == existing
