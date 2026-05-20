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


class TestSubAgentsParallel:

    def test_all_agents_run_and_results_merged(self):
        from src.agents.planner import run_sub_agents_async
        from src.models.planner import PlannerToolResults

        ctx = _make_trip_context()

        transport_result = MagicMock(spec=PlannerToolResults)
        transport_result.raw_results = {"fetch_flights": '{"airline": "El Al", "price": 350}'}

        stay_result = MagicMock(spec=PlannerToolResults)
        stay_result.raw_results = {"fetch_hotels": '{"name": "Ibis", "price_per_night": 80}'}

        experience_result = MagicMock(spec=PlannerToolResults)
        experience_result.raw_results = {"fetch_activities": '{"activity": "Louvre", "price": 20}'}

        with (
            patch("src.agents.planner.TransportAgent") as MockTransport,
            patch("src.agents.planner.StayAgent") as MockStay,
            patch("src.agents.planner.ExperienceAgent") as MockExperience,
        ):
            MockTransport.return_value.run = AsyncMock(return_value=transport_result)
            MockTransport.return_value.agent_name = "transport"
            MockStay.return_value.run = AsyncMock(return_value=stay_result)
            MockStay.return_value.agent_name = "stay"
            MockExperience.return_value.run = AsyncMock(return_value=experience_result)
            MockExperience.return_value.agent_name = "experience"

            merged = asyncio.run(run_sub_agents_async(context=ctx))

        assert "fetch_flights" in merged
        assert "fetch_hotels" in merged
        assert "fetch_activities" in merged

    def test_failed_agent_does_not_block_others(self):
        from src.agents.planner import run_sub_agents_async
        from src.models.planner import PlannerToolResults

        ctx = _make_trip_context()

        stay_result = MagicMock(spec=PlannerToolResults)
        stay_result.raw_results = {"fetch_hotels": '{"name": "Ibis"}'}

        experience_result = MagicMock(spec=PlannerToolResults)
        experience_result.raw_results = {"fetch_activities": '{"activity": "Louvre"}'}

        with (
            patch("src.agents.planner.TransportAgent") as MockTransport,
            patch("src.agents.planner.StayAgent") as MockStay,
            patch("src.agents.planner.ExperienceAgent") as MockExperience,
        ):
            async def _transport_crash(**kw):
                raise RuntimeError("Transport API is down")

            MockTransport.return_value.run = _transport_crash
            MockTransport.return_value.agent_name = "transport"
            MockStay.return_value.run = AsyncMock(return_value=stay_result)
            MockStay.return_value.agent_name = "stay"
            MockExperience.return_value.run = AsyncMock(return_value=experience_result)
            MockExperience.return_value.agent_name = "experience"

            merged = asyncio.run(run_sub_agents_async(context=ctx))

        assert "fetch_flights" not in merged
        assert "fetch_hotels" in merged
        assert "fetch_activities" in merged

    def test_existing_results_are_preserved(self):
        from src.agents.planner import run_sub_agents_async
        from src.models.planner import PlannerToolResults

        ctx = _make_trip_context()
        existing = {"fetch_flights": '{"airline": "El Al", "price": 350}'}

        stay_result = MagicMock(spec=PlannerToolResults)
        stay_result.raw_results = {"fetch_hotels": '{"name": "Ibis"}'}

        empty_result = MagicMock(spec=PlannerToolResults)
        empty_result.raw_results = {}

        with (
            patch("src.agents.planner.TransportAgent") as MockTransport,
            patch("src.agents.planner.StayAgent") as MockStay,
            patch("src.agents.planner.ExperienceAgent") as MockExperience,
        ):
            MockTransport.return_value.run = AsyncMock(return_value=empty_result)
            MockTransport.return_value.agent_name = "transport"
            MockStay.return_value.run = AsyncMock(return_value=stay_result)
            MockStay.return_value.agent_name = "stay"
            MockExperience.return_value.run = AsyncMock(return_value=empty_result)
            MockExperience.return_value.agent_name = "experience"

            merged = asyncio.run(
                run_sub_agents_async(context=ctx, existing_results=existing)
            )

        assert "fetch_flights" in merged
        assert "El Al" in merged["fetch_flights"]
        assert "fetch_hotels" in merged