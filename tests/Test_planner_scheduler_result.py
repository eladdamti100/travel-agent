"""
Tests for planner scheduler result — build_scheduler_result
"""


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


class TestPlannerSchedulerResult:

    def test_ready_tasks_placed_in_wave_1(self):
        from src.agents.planner import build_planner_dependency_graph, build_scheduler_result

        ctx = _make_trip_context()
        graph = build_planner_dependency_graph(ctx, completed_tasks=[])
        scheduler = build_scheduler_result(graph)

        assert len(scheduler.waves) >= 1
        wave_1_tasks = scheduler.waves[0].tasks
        for task in graph.ready_tasks:
            assert task in wave_1_tasks

    def test_blocked_tasks_not_in_any_wave(self):
        from src.agents.planner import build_planner_dependency_graph, build_scheduler_result

        ctx = _make_trip_context()
        graph = build_planner_dependency_graph(ctx, completed_tasks=[])
        scheduler = build_scheduler_result(graph)

        all_scheduled = {task for wave in scheduler.waves for task in wave.tasks}
        for blocked in graph.blocked_tasks:
            assert blocked not in all_scheduled

    def test_empty_graph_produces_no_waves(self):
        from src.agents.planner import build_scheduler_result
        from src.models.planner import PlannerDependencyGraph

        empty_graph = PlannerDependencyGraph(
            nodes={},
            dependencies=[],
            ready_tasks=[],
            blocked_tasks=[],
            completed_tasks=[],
        )
        scheduler = build_scheduler_result(empty_graph)
        assert scheduler.waves == []

    def test_scheduler_tracks_completed_and_blocked(self):
        from src.agents.planner import build_planner_dependency_graph, build_scheduler_result
        from src.models.planner import PlannerTaskType

        ctx = _make_trip_context()
        completed = [PlannerTaskType.FETCH_FLIGHTS]
        graph = build_planner_dependency_graph(ctx, completed_tasks=completed)
        scheduler = build_scheduler_result(graph)

        assert PlannerTaskType.FETCH_FLIGHTS in scheduler.completed_tasks
        assert PlannerTaskType.CALCULATE_TRIP_COST in scheduler.blocked_tasks
