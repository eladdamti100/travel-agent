from unittest.mock import patch, AsyncMock, MagicMock
from src.agents.planner import run_master_planner
from tests.test_cache_check import _make_state
from tests.test_planner_dependency_graph import _make_trip_context

class TestHitlResume:
    @patch("src.agents.planner.enrich_trip_context_async", new_callable=AsyncMock)
    @patch("src.agents.planner.run_sub_agents_async", new_callable=AsyncMock)
    @patch("src.agents.planner.generate_final_plan", new_callable=AsyncMock)
    @patch("src.services.plan_enricher.calculate_cost_if_possible", new_callable=AsyncMock)
    def test_resume_uses_pending_trip_context(self, mock_cost, mock_final, mock_sub, mock_enrich):
        # שימוש ב-AsyncMock מונע את ה-TypeError של הקורוטינה
        mock_enrich.return_value = MagicMock(confidence=1.0)
        mock_final.return_value = "Plan"
        mock_sub.return_value = {"fetch_flights": "", "fetch_hotels": "", "check_visa": "", "fetch_activities": ""}
        mock_cost.return_value = ""
        
        state = _make_state(
            awaiting_user_clarification=True, 
            pending_trip_context=_make_trip_context().model_dump()
        )
        result = run_master_planner(state)
        assert result.get("awaiting_user_clarification") is False
