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
