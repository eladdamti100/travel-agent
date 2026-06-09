"""
Tests for planner dependency graph — build_planner_dependency_graph
"""


def _make_trip_context(**overrides):
    from src.models.trip_context import TripContext
    defaults = dict(
        origin_airport="TLV", origin_country="Israel",
        destination_city="Paris", destination_country="France",
        duration_days=5, total_budget=2000.0,
    )
    defaults.update(overrides)
    return TripContext(**defaults)


class TestPlannerDependencyGraph:

    def test_cost_task_follows_dependency_status(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        ctx = _make_trip_context()
        # No deps completed → BLOCKED
        graph = build_planner_dependency_graph(ctx, completed_tasks=[])
        assert graph.nodes[PlannerTaskType.CALCULATE_TRIP_COST].status == DependencyStatus.BLOCKED
        # Both deps done → READY
        graph = build_planner_dependency_graph(
            ctx, completed_tasks=[PlannerTaskType.FETCH_FLIGHTS, PlannerTaskType.FETCH_HOTELS]
        )
        assert graph.nodes[PlannerTaskType.CALCULATE_TRIP_COST].status == DependencyStatus.READY

    def test_flights_blocked_when_origin_airport_missing(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        graph = build_planner_dependency_graph(_make_trip_context(origin_airport=None))
        node = graph.nodes[PlannerTaskType.FETCH_FLIGHTS]
        assert node.status == DependencyStatus.BLOCKED
        assert "origin_airport" in node.reason

    def test_completed_tasks_marked_correctly(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        graph = build_planner_dependency_graph(
            _make_trip_context(), completed_tasks=[PlannerTaskType.FETCH_FLIGHTS]
        )
        assert graph.nodes[PlannerTaskType.FETCH_FLIGHTS].status == DependencyStatus.COMPLETED
        assert PlannerTaskType.FETCH_FLIGHTS in graph.completed_tasks

    def test_cost_depends_on_flights_and_hotels(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import PlannerTaskType

        graph = build_planner_dependency_graph(_make_trip_context())
        deps = graph.nodes[PlannerTaskType.CALCULATE_TRIP_COST].depends_on
        assert PlannerTaskType.FETCH_FLIGHTS in deps
        assert PlannerTaskType.FETCH_HOTELS in deps

    def test_ready_and_blocked_lists_are_disjoint(self):
        from src.agents.planner import build_planner_dependency_graph

        graph = build_planner_dependency_graph(_make_trip_context())
        assert set(graph.ready_tasks).isdisjoint(set(graph.blocked_tasks))


# ── diff_changed_tasks — Replanner invalidation regression tests ──────────────

class TestDiffChangedTasks:
    """
    Audit Item 3: verify diff_changed_tasks correctly invalidates the exact
    intersection of Tier 1 (DB) and Tier 2 (Web) tasks when context changes,
    WITHOUT invalidating unrelated tasks (e.g. fetch_hotels when only the
    origin_airport changes).
    """

    def _diff(self, old_kwargs: dict, new_kwargs: dict):
        from src.agents.planner_dependencies import diff_changed_tasks
        old = _make_trip_context(**old_kwargs)
        new = _make_trip_context(**new_kwargs)
        return set(diff_changed_tasks(old, new))

    # ── origin_airport change ─────────────────────────────────────────────────

    def test_origin_airport_change_invalidates_flights(self):
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "fetch_flights" in invalidated

    def test_origin_airport_change_invalidates_transport_web_key(self):
        """Tier 2 key transport_live_research must be invalidated with flights."""
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "transport_live_research" in invalidated, (
            "transport_live_research (Tier 2 web key) must be invalidated when "
            "origin_airport changes — it is flight-route dependent"
        )

    def test_origin_airport_change_invalidates_visa(self):
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "check_visa" in invalidated

    def test_origin_airport_change_cascades_to_cost(self):
        """CALCULATE_TRIP_COST depends on FETCH_FLIGHTS → must cascade."""
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "calculate_trip_cost" in invalidated

    def test_origin_airport_change_preserves_hotels(self):
        """fetch_hotels does NOT depend on origin_airport — must NOT be invalidated."""
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "fetch_hotels" not in invalidated, (
            "fetch_hotels must be preserved (reused from cache) when only the "
            "origin_airport changes — hotels are destination-only"
        )

    def test_origin_airport_change_preserves_stay_web_key(self):
        """stay_live_research is hotel-only — must be preserved on airport change."""
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "stay_live_research" not in invalidated

    def test_origin_airport_change_preserves_activities(self):
        invalidated = self._diff({"origin_airport": "TLV"}, {"origin_airport": "JFK"})
        assert "fetch_activities" not in invalidated

    # ── destination_city change → full replan ─────────────────────────────────

    def test_destination_change_triggers_full_replan(self):
        from src.models.planner import PlannerTaskType
        from src.agents.planner_dependencies import _WEB_AGENT_EXTRA_KEYS

        invalidated = self._diff(
            {"destination_city": "Paris"},
            {"destination_city": "Tokyo"},
        )
        # Every PlannerTaskType value must be present.
        for task in PlannerTaskType:
            assert task.value in invalidated, (
                f"{task.value} missing from full-replan invalidation set"
            )
        # All web-agent extra keys must also be present.
        for key in _WEB_AGENT_EXTRA_KEYS:
            assert key in invalidated, (
                f"Web-agent key '{key}' missing from full-replan invalidation set"
            )

    # ── total_budget change ───────────────────────────────────────────────────

    def test_budget_change_invalidates_cost_and_currency(self):
        invalidated = self._diff({"total_budget": 2000.0}, {"total_budget": 5000.0})
        assert "calculate_trip_cost" in invalidated
        assert "live_currency_conversion" in invalidated

    def test_budget_change_preserves_flights_and_hotels(self):
        """A budget change never requires re-fetching flights or hotels."""
        invalidated = self._diff({"total_budget": 2000.0}, {"total_budget": 5000.0})
        assert "fetch_flights" not in invalidated
        assert "fetch_hotels" not in invalidated

    # ── no change ────────────────────────────────────────────────────────────

    def test_identical_contexts_return_empty_list(self):
        invalidated = self._diff({}, {})
        assert invalidated == set(), (
            "No context fields changed → nothing should be invalidated"
        )

    # ── task_registry dual-format normalization ───────────────────────────────

    def test_registry_resolves_string_key(self):
        """String task keys (e.g. from JSON checkpoint) must resolve without KeyError."""
        from src.agents.task_registry import get_agents_for_tasks
        agents = get_agents_for_tasks({"fetch_flights"})
        assert len(agents) == 1
        assert agents[0].agent_name == "transport_agent"

    def test_registry_resolves_enum_key(self):
        """Enum task keys (from in-memory planner DAG) must resolve without KeyError."""
        from src.agents.task_registry import get_agents_for_tasks
        from src.models.planner import PlannerTaskType
        agents = get_agents_for_tasks({PlannerTaskType.FETCH_FLIGHTS})
        assert len(agents) == 1
        assert agents[0].agent_name == "transport_agent"

    def test_all_task_types_in_requirements_map(self):
        """Every PlannerTaskType must have an entry in _TASK_REQUIREMENTS."""
        from src.agents.planner_dependencies import _TASK_REQUIREMENTS
        from src.models.planner import PlannerTaskType
        missing = [t for t in PlannerTaskType if t not in _TASK_REQUIREMENTS]
        assert not missing, (
            f"PlannerTaskType members missing from _TASK_REQUIREMENTS: {missing}"
        )
