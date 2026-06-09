from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
from src.agents.planner import run_master_planner
from src.models.final_plan import FinalPlan


def _make_trip_context(**overrides):
    from src.models.trip_context import TripContext
    defaults = dict(
        origin_airport="TLV", origin_country="Israel",
        destination_city="Paris", destination_country="France",
        duration_days=5, total_budget=2000.0,
    )
    defaults.update(overrides)
    return TripContext(**defaults)


def _make_state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="Plan a trip to Paris for 5 days from TLV, Israeli passport, $2000")],
        "cache_status": "miss",
        "cache_similarity_score": 0.0,
        "cache_matched_query": "",
        "cache_answer": "",
    }
    base.update(overrides)
    return base


class TestHitlResume:
    @patch("src.agents.planner.enrich_trip_context_async", new_callable=AsyncMock)
    @patch("src.agents.planner.run_sub_agents_async", new_callable=AsyncMock)
    @patch("src.agents.planner.generate_final_plan", new_callable=AsyncMock)
    @patch("src.services.plan_enricher.calculate_cost_if_possible", new_callable=AsyncMock)
    @patch("src.services.plan_enricher.fill_missing_with_web", new_callable=AsyncMock)
    def test_resume_uses_pending_trip_context(
        self, mock_web_fill, mock_cost, mock_final, mock_sub, mock_enrich
    ):
        # generate_final_plan returns Tuple[str, FinalPlan] — must be a 2-tuple.
        mock_enrich.return_value = MagicMock(confidence=1.0, preference_updates=[])
        mock_final.return_value = ("Here is your Paris plan.", MagicMock(spec=FinalPlan))
        mock_sub.return_value = {
            "fetch_flights": "[]", "fetch_hotels": "[]",
            "check_visa": "no visa", "fetch_activities": "[]",
        }
        mock_cost.return_value = ""
        mock_web_fill.return_value = mock_sub.return_value

        state = _make_state(
            awaiting_user_clarification=True,
            pending_trip_context=_make_trip_context().model_dump(),
        )
        result = run_master_planner(state)
        # After resume the planner must clear the HITL pause flag.
        assert result.get("awaiting_user_clarification") is False

    @patch("src.agents.planner.enrich_trip_context_async", new_callable=AsyncMock)
    @patch("src.agents.planner.run_sub_agents_async", new_callable=AsyncMock)
    @patch("src.agents.planner.generate_final_plan", new_callable=AsyncMock)
    @patch("src.services.plan_enricher.calculate_cost_if_possible", new_callable=AsyncMock)
    @patch("src.services.plan_enricher.fill_missing_with_web", new_callable=AsyncMock)
    def test_resume_merges_pending_context_fields(
        self, mock_web_fill, mock_cost, mock_final, mock_sub, mock_enrich
    ):
        """
        Pending context (no origin_airport) + new message providing TLV must
        result in a merged TripContext where origin_airport == 'TLV'.
        """
        mock_enrich.return_value = MagicMock(confidence=1.0, preference_updates=[])
        mock_final.return_value = ("Here is your plan.", MagicMock(spec=FinalPlan))
        mock_sub.return_value = {
            "fetch_flights": "[]", "fetch_hotels": "[]",
            "check_visa": "no visa", "fetch_activities": "[]",
        }
        mock_cost.return_value = ""
        mock_web_fill.return_value = mock_sub.return_value

        pending = _make_trip_context(origin_airport=None).model_dump()
        state = _make_state(
            messages=[HumanMessage(content="my airport is TLV")],
            awaiting_user_clarification=True,
            pending_trip_context=pending,
        )

        from src.graph.nodes import resume_hitl_context_node
        merged_state = {**state, **resume_hitl_context_node(state)}

        # After resume_hitl_context_node the trip_context should carry TLV.
        tc = merged_state.get("trip_context", {})
        assert tc.get("origin_airport") == "TLV"
        assert merged_state.get("awaiting_user_clarification") is False
