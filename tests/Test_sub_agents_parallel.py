"""
Tests for parallel sub-agent execution — run_sub_agents_async
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


def _make_trip_context(**overrides):
    from src.models.trip_context import TripContext
    defaults = dict(
        origin_airport="TLV",
        origin_country="Israel",
        destination_city="Paris",
        destination_country="France",
        duration_days=5,
        total_budget=2000.0,
    )
    defaults.update(overrides)
    return TripContext(**defaults)


def _mock_transport(run_return=None, run_side_effect=None):
    m = MagicMock()
    m.result_keys = ("fetch_flights", "check_visa")
    m.agent_name = "transport"
    if run_side_effect:
        m.run = run_side_effect
    else:
        m.run = AsyncMock(return_value=run_return)
    return m


def _mock_stay(run_return=None):
    m = MagicMock()
    m.result_keys = ("fetch_hotels",)
    m.agent_name = "stay"
    m.run = AsyncMock(return_value=run_return)
    return m


def _mock_experience(run_return=None):
    m = MagicMock()
    m.result_keys = ("fetch_activities",)
    m.agent_name = "experience"
    m.run = AsyncMock(return_value=run_return)
    return m


def _make_result(raw: dict):
    from src.models.planner import PlannerToolResults
    r = MagicMock(spec=PlannerToolResults)
    r.raw_results = raw
    return r


class TestSubAgentsParallel:

    def test_all_agents_run_and_results_merged(self):
        from src.agents.planner import run_sub_agents_async

        ctx = _make_trip_context()
        transport = _mock_transport(_make_result({"fetch_flights": '{"airline": "El Al", "price": 350}'}))
        stay = _mock_stay(_make_result({"fetch_hotels": '{"name": "Ibis", "price_per_night": 80}'}))
        experience = _mock_experience(_make_result({"fetch_activities": '{"activity": "Louvre", "price": 20}'}))

        with (
            patch("src.agents.planner.TransportAgent", return_value=transport),
            patch("src.agents.planner.StayAgent", return_value=stay),
            patch("src.agents.planner.ExperienceAgent", return_value=experience),
        ):
            merged = asyncio.run(run_sub_agents_async(context=ctx))

        assert "fetch_flights" in merged
        assert "fetch_hotels" in merged
        assert "fetch_activities" in merged

    def test_failed_agent_does_not_block_others(self):
        from src.agents.planner import run_sub_agents_async

        ctx = _make_trip_context()

        async def _crash(**kw):
            raise RuntimeError("Transport API is down")

        transport = _mock_transport(run_side_effect=_crash)
        stay = _mock_stay(_make_result({"fetch_hotels": '{"name": "Ibis"}'}))
        experience = _mock_experience(_make_result({"fetch_activities": '{"activity": "Louvre"}'}))

        with (
            patch("src.agents.planner.TransportAgent", return_value=transport),
            patch("src.agents.planner.StayAgent", return_value=stay),
            patch("src.agents.planner.ExperienceAgent", return_value=experience),
        ):
            merged = asyncio.run(run_sub_agents_async(context=ctx))

        assert "fetch_flights" not in merged
        assert "fetch_hotels" in merged
        assert "fetch_activities" in merged

    def test_existing_results_are_preserved(self):
        from src.agents.planner import run_sub_agents_async

        ctx = _make_trip_context()
        existing = {"fetch_flights": '{"airline": "El Al", "price": 350}'}

        transport = _mock_transport(_make_result({}))
        stay = _mock_stay(_make_result({"fetch_hotels": '{"name": "Ibis"}'}))
        experience = _mock_experience(_make_result({}))

        with (
            patch("src.agents.planner.TransportAgent", return_value=transport),
            patch("src.agents.planner.StayAgent", return_value=stay),
            patch("src.agents.planner.ExperienceAgent", return_value=experience),
        ):
            merged = asyncio.run(
                run_sub_agents_async(context=ctx, existing_results=existing)
            )

        assert "fetch_flights" in merged
        assert "El Al" in merged["fetch_flights"]
        assert "fetch_hotels" in merged

    def test_agent_skipped_when_all_result_keys_covered(self):
        """TransportAgent.run must not be called when all its result_keys are in existing_results."""
        from src.agents.planner import run_sub_agents_async

        ctx = _make_trip_context()
        existing = {
            "fetch_flights": '{"airline": "El Al"}',
            "check_visa": "No visa required.",
        }

        transport = _mock_transport(_make_result({}))
        stay = _mock_stay(_make_result({"fetch_hotels": '{"name": "Ibis"}'}))
        experience = _mock_experience(_make_result({"fetch_activities": '{"activity": "Louvre"}'}))

        with (
            patch("src.agents.planner.TransportAgent", return_value=transport),
            patch("src.agents.planner.StayAgent", return_value=stay),
            patch("src.agents.planner.ExperienceAgent", return_value=experience),
        ):
            merged = asyncio.run(
                run_sub_agents_async(context=ctx, existing_results=existing)
            )

        transport.run.assert_not_awaited()
        assert "fetch_flights" in merged
        assert "fetch_hotels" in merged
        assert "fetch_activities" in merged

    def test_all_agents_skipped_when_fully_covered(self):
        """No agent.run() is called when existing_results covers every result_key."""
        from src.agents.planner import run_sub_agents_async

        ctx = _make_trip_context()
        existing = {
            "fetch_flights": "...",
            "check_visa": "...",
            "fetch_hotels": "...",
            "fetch_activities": "...",
        }

        transport = _mock_transport(_make_result({}))
        stay = _mock_stay(_make_result({}))
        experience = _mock_experience(_make_result({}))

        with (
            patch("src.agents.planner.TransportAgent", return_value=transport),
            patch("src.agents.planner.StayAgent", return_value=stay),
            patch("src.agents.planner.ExperienceAgent", return_value=experience),
        ):
            merged = asyncio.run(
                run_sub_agents_async(context=ctx, existing_results=existing)
            )

        transport.run.assert_not_awaited()
        stay.run.assert_not_awaited()
        experience.run.assert_not_awaited()
        assert merged == existing
