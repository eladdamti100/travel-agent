"""
Tests for planner dependency graph — build_planner_dependency_graph
"""

import pytest


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


class TestPlannerDependencyGraph:

    def test_cost_ready_when_flights_and_hotels_completed(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        ctx = _make_trip_context()
        completed = [
            PlannerTaskType.FETCH_FLIGHTS,
            PlannerTaskType.FETCH_HOTELS,
        ]
        graph = build_planner_dependency_graph(ctx, completed_tasks=completed)

        cost_node = graph.nodes[PlannerTaskType.CALCULATE_TRIP_COST]
        assert cost_node.status == DependencyStatus.READY

    def test_cost_blocked_without_completed_dependencies(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        ctx = _make_trip_context()
        graph = build_planner_dependency_graph(ctx, completed_tasks=[])

        cost_node = graph.nodes[PlannerTaskType.CALCULATE_TRIP_COST]
        assert cost_node.status == DependencyStatus.BLOCKED

    def test_flights_blocked_when_origin_airport_missing(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        ctx = _make_trip_context(origin_airport=None)
        graph = build_planner_dependency_graph(ctx)

        flights_node = graph.nodes[PlannerTaskType.FETCH_FLIGHTS]
        assert flights_node.status == DependencyStatus.BLOCKED
        assert "origin_airport" in flights_node.reason

    def test_completed_tasks_marked_correctly(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import DependencyStatus, PlannerTaskType

        ctx = _make_trip_context()
        graph = build_planner_dependency_graph(
            ctx, completed_tasks=[PlannerTaskType.FETCH_FLIGHTS]
        )

        assert graph.nodes[PlannerTaskType.FETCH_FLIGHTS].status == DependencyStatus.COMPLETED
        assert PlannerTaskType.FETCH_FLIGHTS in graph.completed_tasks

    def test_cost_depends_on_flights_and_hotels(self):
        from src.agents.planner import build_planner_dependency_graph
        from src.models.planner import PlannerTaskType

        ctx = _make_trip_context()
        graph = build_planner_dependency_graph(ctx)

        cost_node = graph.nodes[PlannerTaskType.CALCULATE_TRIP_COST]
        assert PlannerTaskType.FETCH_FLIGHTS in cost_node.depends_on
        assert PlannerTaskType.FETCH_HOTELS in cost_node.depends_on

    def test_ready_and_blocked_lists_are_disjoint(self):
        from src.agents.planner import build_planner_dependency_graph

        ctx = _make_trip_context()
        graph = build_planner_dependency_graph(ctx)

        assert set(graph.ready_tasks).isdisjoint(set(graph.blocked_tasks))