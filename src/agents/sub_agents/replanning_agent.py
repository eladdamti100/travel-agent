"""
Replanning agent.

Handles partial trip modifications without rebuilding the entire trip from scratch.

The replanning agent:
- compares old and new TripContext values
- determines which planner tasks became invalid
- decides which sub-agents/tasks must rerun
- preserves unaffected planner results
"""

from dataclasses import dataclass
from typing import Dict, List

from src.agents.planner_dependencies import diff_changed_tasks
from src.models.planner import PlannerTaskType
from src.models.trip_context import TripContext
from src.utils.logger import get_logger

logger = get_logger("replanning_agent")


@dataclass
class ReplanningResult:
    """
    Result of replanning analysis.
    """

    changed_tasks: List[PlannerTaskType]
    preserved_results: Dict[str, str]
    invalidated_results: Dict[str, str]
    hitl_feedback: str = ""


def analyze_replanning(
    *,
    old_context: TripContext,
    new_context: TripContext,
    existing_results: Dict[str, str],
    hitl_feedback: str = "",
) -> ReplanningResult:
    """
    Determines which planner tasks must rerun after user modifications.
    """
    logger.info(
        "Replanning analysis started. old_context=%s new_context=%s",
        old_context.model_dump(),
        new_context.model_dump(),
    )

    changed_tasks = diff_changed_tasks(
        old_context=old_context,
        new_context=new_context,
    )

    if not changed_tasks:
        logger.info(
            "Replanning found no task-level changes; preserving all existing results."
        )

        return ReplanningResult(
            changed_tasks=[],
            preserved_results=dict(existing_results),
            invalidated_results={},
            hitl_feedback=hitl_feedback,
        )

    invalidated_keys = {task.value for task in changed_tasks}

    logger.info(
    "Replanning invalidated task keys=%s",
    list(invalidated_keys),
    )  

    preserved_results = {
        key: value
        for key, value in existing_results.items()
        if key not in invalidated_keys
    }

    invalidated_results = {
        key: value
        for key, value in existing_results.items()
        if key in invalidated_keys
    }

    logger.info(
        "Replanning analysis completed. changed=%s preserved=%s invalidated=%s",
        [task.value for task in changed_tasks],
        list(preserved_results.keys()),
        list(invalidated_results.keys()),
    )

    return ReplanningResult(
        changed_tasks=changed_tasks,
        preserved_results=preserved_results,
        invalidated_results=invalidated_results,
        hitl_feedback=hitl_feedback,
    )