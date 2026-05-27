"""
Planner scheduler.

Answers: "in what order / waves should ready tasks execute?"
"""

from typing import Dict, List

from src.models.planner import (
    PlannerDependencyGraph,
    PlannerTaskType,
    SchedulerResult,
    SchedulerWave,
)

_PLANNER_TASK_VALUE_SET: frozenset = frozenset(item.value for item in PlannerTaskType)


def build_scheduler_result(
    dependency_graph: PlannerDependencyGraph,
) -> SchedulerResult:
    """
    Builds async execution waves from the dependency graph.

    Tasks in the same wave can run in parallel.
    """
    waves: List[SchedulerWave] = []

    if dependency_graph.ready_tasks:
        waves.append(
            SchedulerWave(
                wave_number=1,
                tasks=dependency_graph.ready_tasks,
                reason=(
                    "Tasks with all required context fields "
                    "and completed dependencies."
                ),
            )
        )

    wave_debug = [
        {
            "wave": wave.wave_number,
            "tasks": [task.value for task in wave.tasks],
        }
        for wave in waves
    ]

    return SchedulerResult(
        waves=waves,
        completed_tasks=dependency_graph.completed_tasks,
        blocked_tasks=dependency_graph.blocked_tasks,
        reason=(
            "Scheduler grouped dependency-ready planner tasks "
            f"into async execution waves: {wave_debug}"
        ),
    )

def completed_tasks_from(task_results: Dict[str, str]) -> List[PlannerTaskType]:
    """Returns PlannerTaskType values for every key present in task_results."""
    return [
        PlannerTaskType(key)
        for key in task_results
        if key in _PLANNER_TASK_VALUE_SET
    ]
