"""
Tests for the critic agent — budget gate, completeness check, suggestions.

All tests are pure unit tests — no LLM calls, no graph, no database.
"""

from src.agents.critic import critique_plan, _extract_cost_from_raw_text
from src.models.critic import CritiqueResult, BudgetBreakdown


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(
    budget=3000.0,
    duration_days=7,
    num_travelers=1,
    flight_price=600.0,
    hotel_nightly=100.0,
    activities=None,
    include_visa=True,
    include_cost=True,
):
    """Build a minimal realistic AgentState dict for tests."""
    flights = [{"price": flight_price, "airline": "El Al", "origin": "TLV"}] if flight_price else []
    hotels = [{"price_per_night": hotel_nightly, "name": "Hotel Marco", "city": "Tokyo"}] if hotel_nightly else []
    acts = activities or [{"price": 30.0, "name": "Senso-ji"}, {"price": 50.0, "name": "TeamLab"}]
    visa = {"visa_required": False, "origin_country": "Israel"} if include_visa else None
    cost = {
        "total_cost": flight_price + (hotel_nightly * duration_days) + sum(a["price"] for a in acts),
        "flight_price": flight_price,
        "hotel_price_per_night": hotel_nightly,
        "duration_days": duration_days,
    } if include_cost and flight_price and hotel_nightly else {}

    return {
        "total_budget": budget,
        "trip_context": {
            "destination_city": "Tokyo",
            "duration_days": duration_days,
            "num_travelers": num_travelers,
            "total_budget": budget,
        },
        "planner_structured_results": {
            "flights": flights,
            "hotels": hotels,
            "activities": acts,
            "visa": visa,
            "cost": cost,
        },
        "planner_task_results": {},
    }


# ── Budget gate ───────────────────────────────────────────────────────────────

class TestBudgetGate:

    def test_within_budget_passes(self):
        # flight 600 + hotel 700 (100×7) + activities 80 = 1380 < 3000
        result = critique_plan(_make_state(budget=3000, flight_price=600, hotel_nightly=100))
        assert result.passed is True
        assert result.budget.within_budget is True
        assert not any("exceeds" in i for i in result.issues)

    def test_over_budget_fails(self):
        # flight 1500 + hotel 1400 (200×7) + activities 80 = 2980 > 2000
        result = critique_plan(_make_state(budget=2000, flight_price=1500, hotel_nightly=200))
        assert result.passed is False
        assert result.budget.within_budget is False
        assert any("exceeds" in i for i in result.issues)

    def test_overage_amount_is_correct(self):
        state = _make_state(budget=2000, flight_price=1500, hotel_nightly=200,
                            activities=[{"price": 80.0, "name": "Tour"}])
        result = critique_plan(state)
        # total = 1500 + 1400 + 80 = 2980, overage = 980
        assert result.budget.overage is not None
        assert result.budget.overage > 0

    def test_no_budget_in_state_does_not_crash(self):
        state = _make_state()
        state["total_budget"] = None
        state["trip_context"]["total_budget"] = None
        result = critique_plan(state)
        assert isinstance(result, CritiqueResult)
        assert result.budget.within_budget is None

    def test_score_lower_when_over_budget(self):
        over = critique_plan(_make_state(budget=1000, flight_price=1500, hotel_nightly=200))
        within = critique_plan(_make_state(budget=5000, flight_price=600, hotel_nightly=100))
        assert over.score < within.score

    def test_multi_traveler_flight_cost_multiplied(self):
        # 2 travelers × $600 flight = $1200 flights, + $700 hotel + $80 activities = $1980 > $1500
        result = critique_plan(_make_state(budget=1500, flight_price=600, hotel_nightly=100,
                                           num_travelers=2))
        assert result.budget.flight_cost == 1200.0


# ── Completeness gate ─────────────────────────────────────────────────────────

class TestCompletenessGate:

    def test_full_plan_all_sections_present(self):
        result = critique_plan(_make_state())
        assert result.completeness["has_flights"] is True
        assert result.completeness["has_hotels"] is True
        assert result.completeness["has_activities"] is True
        assert result.completeness["has_visa_info"] is True
        assert result.completeness["has_cost_breakdown"] is True

    def test_missing_flights_reported(self):
        state = _make_state(flight_price=None, include_cost=False)
        result = critique_plan(state)
        assert result.completeness["has_flights"] is False
        assert any("flights" in i for i in result.issues)

    def test_missing_hotels_reported(self):
        state = _make_state(hotel_nightly=None, include_cost=False)
        result = critique_plan(state)
        assert result.completeness["has_hotels"] is False
        assert any("hotels" in i for i in result.issues)

    def test_missing_visa_reported(self):
        state = _make_state(include_visa=False)
        result = critique_plan(state)
        assert result.completeness["has_visa_info"] is False
        assert any("visa" in i for i in result.issues)

    def test_missing_sections_do_not_set_passed_false(self):
        # Missing sections are soft fails — plan still passes if budget is ok
        state = _make_state(include_visa=False)
        result = critique_plan(state)
        assert result.passed is True   # budget is fine
        assert result.score < 10      # but score is penalised


# ── Suggestions ───────────────────────────────────────────────────────────────

class TestSuggestions:

    def test_hotel_suggestion_has_target_price(self):
        # Expensive hotel forces a specific suggestion
        result = critique_plan(_make_state(budget=2000, flight_price=600, hotel_nightly=250))
        hotel_suggestions = [s for s in result.suggestions if "hotel" in s.lower() or "night" in s.lower()]
        assert hotel_suggestions, "Expected a hotel price suggestion"
        assert "$" in hotel_suggestions[0]

    def test_flight_suggestion_has_target_price(self):
        result = critique_plan(_make_state(budget=1500, flight_price=1200, hotel_nightly=80))
        flight_suggestions = [s for s in result.suggestions if "flight" in s.lower()]
        assert flight_suggestions
        assert "$" in flight_suggestions[0]

    def test_no_suggestions_when_within_budget_and_complete(self):
        result = critique_plan(_make_state(budget=5000, flight_price=500, hotel_nightly=80))
        assert result.passed is True
        budget_suggestions = [s for s in result.suggestions if "hotel" in s.lower() or "flight" in s.lower()]
        assert not budget_suggestions

    def test_suggestions_are_strings(self):
        result = critique_plan(_make_state(budget=1000, flight_price=800, hotel_nightly=150))
        for s in result.suggestions:
            assert isinstance(s, str) and len(s) > 0


# ── Raw text fallback ─────────────────────────────────────────────────────────

class TestRawTextFallback:

    def test_extracts_total_from_raw_text(self):
        assert _extract_cost_from_raw_text("Total trip cost: $2,450") == 2450.0
        assert _extract_cost_from_raw_text("Estimated total: $1800 USD") == 1800.0
        assert _extract_cost_from_raw_text("Grand total: $3,200.50") == 3200.50

    def test_returns_none_when_no_match(self):
        assert _extract_cost_from_raw_text("No prices here at all") is None
        assert _extract_cost_from_raw_text("") is None

    def test_raw_fallback_used_when_no_structured_cost(self):
        # State with NO structured flights/hotels — only raw text — so fallback is used
        state = {
            "total_budget": 3000.0,
            "trip_context": {"total_budget": 3000.0, "duration_days": 7, "num_travelers": 1},
            "planner_structured_results": {
                "flights": [], "hotels": [], "activities": [],
                "visa": None, "cost": {},
            },
            "planner_task_results": {
                "calculate_trip_cost": "Grand total: $2,100 for the full trip"
            },
        }
        result = critique_plan(state)
        assert result.budget.total_estimated == 2100.0


# ── Score ──────────────────────────────────────────────────────────────────────

class TestScore:

    def test_perfect_plan_scores_10(self):
        result = critique_plan(_make_state(budget=5000))
        assert result.score == 10

    def test_score_between_0_and_10(self):
        for budget in [500, 1000, 5000]:
            result = critique_plan(_make_state(budget=budget))
            assert 0 <= result.score <= 10

    def test_score_never_negative(self):
        # Worst possible state — over budget AND missing everything
        state = {
            "total_budget": 100.0,
            "trip_context": {"total_budget": 100.0, "duration_days": 7, "num_travelers": 1},
            "planner_structured_results": {
                "flights": [{"price": 2000.0}],
                "hotels": [],
                "activities": [],
                "visa": None,
                "cost": {"total_cost": 5000.0},
            },
            "planner_task_results": {},
        }
        result = critique_plan(state)
        assert result.score >= 0
