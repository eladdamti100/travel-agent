"""
Planner dependency analysis.

Answers: "what can run, what is blocked, what is missing?"
"""

from typing import Dict, List, Optional, Tuple

from src.models.planner import (
    DependencyCheckResult,
    DependencyStatus,
    MissingRequirement,
    PlannerDependency,
    PlannerDependencyGraph,
    PlannerStatus,
    PlannerTask,
    PlannerTaskNode,
    PlannerTaskStatus,
    PlannerTaskType,
)
from src.models.trip_context import REQUIRED_TRIP_FIELDS, TripContext

_PLANNER_TASK_VALUE_SET: frozenset = frozenset(item.value for item in PlannerTaskType)

_TASK_REQUIREMENTS: Dict[PlannerTaskType, Tuple[str, ...]] = {
    PlannerTaskType.FETCH_FLIGHTS: ("origin_airport", "destination_city"),
    PlannerTaskType.FETCH_HOTELS: ("destination_city",),
    PlannerTaskType.FETCH_ACTIVITIES: ("destination_city",),
    PlannerTaskType.CHECK_VISA: ("origin_country", "destination_country"),

    # Cost calculation is special: it also depends on tool results
    # from fetch_flights and fetch_hotels. Those are checked later in
    # _calculate_cost_if_possible().
    PlannerTaskType.CALCULATE_TRIP_COST: ("duration_days",),

    PlannerTaskType.FETCH_RESTAURANTS: ("destination_city",),
    PlannerTaskType.LOCAL_TRANSPORT_GUIDE: ("destination_city",),
    PlannerTaskType.FETCH_WEATHER: ("destination_city", "travel_month"),
    PlannerTaskType.EVENTS_FINDER: ("destination_city", "travel_month"),
    PlannerTaskType.AIRPORT_TRANSFER_INFO: ("destination_city",),

    # Web intelligence tasks — executed by WebAgent
    PlannerTaskType.GEOCODE_LOCATION: ("destination_city",),
    PlannerTaskType.FETCH_LIVE_EVENTS: ("destination_city",),
    PlannerTaskType.LIVE_CURRENCY_CONVERSION: (),
    PlannerTaskType.FETCH_BREWERIES: ("destination_city",),
    PlannerTaskType.FETCH_COUNTRY_METADATA: ("destination_city",),
    PlannerTaskType.WEB_RESEARCH_TAVILY: ("destination_city",),
}


def build_planner_dependency_graph(
    context: TripContext,
    completed_tasks: Optional[List[PlannerTaskType]] = None,
) -> PlannerDependencyGraph:
    """
    Builds an explicit planner dependency DAG.
    """
    completed = completed_tasks or []

    dependencies = [
        PlannerDependency(
            task=PlannerTaskType.CALCULATE_TRIP_COST,
            depends_on=PlannerTaskType.FETCH_FLIGHTS,
            reason="Cost calculation needs flight pricing.",
        ),
        PlannerDependency(
            task=PlannerTaskType.CALCULATE_TRIP_COST,
            depends_on=PlannerTaskType.FETCH_HOTELS,
            reason="Cost calculation needs hotel pricing.",
        ),
    ]

    dependency_map: Dict[PlannerTaskType, List[PlannerTaskType]] = {}

    for dependency in dependencies:
        dependency_map.setdefault(dependency.task, []).append(
            dependency.depends_on
        )

    nodes: Dict[PlannerTaskType, PlannerTaskNode] = {}
    ready_tasks: List[PlannerTaskType] = []
    blocked_tasks: List[PlannerTaskType] = []

    for task_type in PlannerTaskType:
        required_fields = list(_TASK_REQUIREMENTS.get(task_type, ()))

        missing_fields = [
            field_name
            for field_name in required_fields
            if getattr(context, field_name) in (None, "", [])
        ]

        task_dependencies = dependency_map.get(task_type, [])

        missing_dependencies = [
            dep
            for dep in task_dependencies
            if dep not in completed
        ]

        if task_type in completed:
            status = DependencyStatus.COMPLETED
            reason = "Task already completed."

        elif missing_fields:
            status = DependencyStatus.BLOCKED
            reason = (
                "Task is blocked by missing context fields: "
                f"{', '.join(missing_fields)}."
            )

        elif missing_dependencies:
            status = DependencyStatus.BLOCKED
            reason = (
                "Task is blocked by unfinished dependencies: "
                + ", ".join(dep.value for dep in missing_dependencies)
                + "."
            )

        else:
            status = DependencyStatus.READY
            reason = "Task has all context fields and dependencies ready."

        node = PlannerTaskNode(
            task_type=task_type,
            status=status,
            required_context_fields=required_fields,
            depends_on=task_dependencies,
            unlocked_by=[
                dependency.task
                for dependency in dependencies
                if dependency.depends_on == task_type
            ],
            reason=reason,
        )

        nodes[task_type] = node

        if status == DependencyStatus.READY:
            ready_tasks.append(task_type)

        elif status == DependencyStatus.BLOCKED:
            blocked_tasks.append(task_type)

    return PlannerDependencyGraph(
        nodes=nodes,
        dependencies=dependencies,
        ready_tasks=ready_tasks,
        blocked_tasks=blocked_tasks,
        completed_tasks=completed,
    )


def check_planner_dependencies(context: TripContext) -> DependencyCheckResult:
    """
    Checks required trip fields and determines which planner tasks can run now.
    """
    missing_requirements = _build_missing_requirements(context)
    tasks = [_build_planner_task(context, task_type) for task_type in PlannerTaskType]

    ready_tasks = [task for task in tasks if task.is_ready]
    blocked_tasks = [task for task in tasks if not task.is_ready]
    async_ready_tasks = [
        task
        for task in ready_tasks
        if task.task_type != PlannerTaskType.CALCULATE_TRIP_COST
    ]

    if not missing_requirements:
        status = PlannerStatus.READY
    elif ready_tasks:
        status = PlannerStatus.PARTIAL_READY
    else:
        status = PlannerStatus.MISSING_REQUIRED_INFO

    hitl_question = (
        _build_hitl_question(missing_requirements)
        if missing_requirements
        else None
    )

    return DependencyCheckResult(
        status=status,
        missing_requirements=missing_requirements,
        ready_tasks=ready_tasks,
        blocked_tasks=blocked_tasks,
        async_ready_tasks=async_ready_tasks,
        hitl_question=hitl_question,
    )


def _build_missing_requirements(context: TripContext) -> List[MissingRequirement]:
    missing = []

    for field_name in REQUIRED_TRIP_FIELDS:
        value = getattr(context, field_name)

        if value not in (None, "", []):
            continue

        missing.append(
            MissingRequirement(
                field_name=field_name,
                reason=_missing_field_reason(field_name),
                user_question=_missing_field_question(field_name),
            )
        )

    return missing


def _build_planner_task(
    context: TripContext,
    task_type: PlannerTaskType,
) -> PlannerTask:
    required_fields = _TASK_REQUIREMENTS.get(task_type, ())
    missing_fields = [
        field_name
        for field_name in required_fields
        if getattr(context, field_name) in (None, "", [])
    ]

    is_ready = not missing_fields

    status = (
        PlannerTaskStatus.READY
        if is_ready
        else PlannerTaskStatus.BLOCKED
    )

    return PlannerTask(
        task_type=task_type,
        status=status,
        is_ready=is_ready,
        missing_fields=missing_fields,
        can_run_async=is_ready and task_type != PlannerTaskType.CALCULATE_TRIP_COST,
        reason=(
            "Task has all required inputs and can run now."
            if is_ready
            else f"Task is missing required fields: {', '.join(missing_fields)}."
        ),
    )


def diff_changed_tasks(
    old_context: TripContext,
    new_context: TripContext,
) -> List[PlannerTaskType]:
    """
    Returns every task that must be re-run when context changes, including
    cascade invalidation through the dependency graph.

    Two reasons a task is invalidated:
      1. Direct  — one of its own required context fields changed.
      2. Cascade — it depends on another task that was directly invalidated
                   (e.g. CALCULATE_TRIP_COST depends on FETCH_FLIGHTS, so if
                   FETCH_FLIGHTS is re-run its downstream tasks must also re-run).

    Fields that did not change → their tasks stay cached (no re-run).

    Examples:
        origin_airport changed   → FETCH_FLIGHTS (direct)
                                 → CALCULATE_TRIP_COST (cascade: needs new flight price)

        destination_city changed → FETCH_FLIGHTS, FETCH_HOTELS, FETCH_ACTIVITIES (direct)
                                 → CALCULATE_TRIP_COST (cascade)

        total_budget changed     → no task has total_budget as a required field,
                                   so nothing is re-run (budget is used only in
                                   the final answer prompt, not in tool calls)
    """
    # Step 1 — find which context fields actually changed
    all_tracked_fields: set = set()
    for fields in _TASK_REQUIREMENTS.values():
        all_tracked_fields.update(fields)

    changed_fields = {
        field
        for field in all_tracked_fields
        if getattr(old_context, field, None) != getattr(new_context, field, None)
    }

    if not changed_fields:
        return []

    # Step 2 — find tasks directly invalidated by changed fields
    directly_invalidated: set = {
        task_type
        for task_type, required_fields in _TASK_REQUIREMENTS.items()
        if any(f in changed_fields for f in required_fields)
    }

    # Step 3 — cascade: any task that depends_on an invalidated task is also
    # invalidated, regardless of whether its own context fields changed.
    # Build a map: task → list of tasks it depends on (from _TASK_DEPENDENCIES).
    _TASK_DEPENDENCIES: Dict[PlannerTaskType, List[PlannerTaskType]] = {
        PlannerTaskType.CALCULATE_TRIP_COST: [
            PlannerTaskType.FETCH_FLIGHTS,
            PlannerTaskType.FETCH_HOTELS,
        ],
    }

    cascaded: set = set(directly_invalidated)
    changed = True
    while changed:
        changed = False
        for task, deps in _TASK_DEPENDENCIES.items():
            if task not in cascaded and any(d in cascaded for d in deps):
                cascaded.add(task)
                changed = True

    return list(cascaded)


def _build_hitl_question(missing_requirements: List[MissingRequirement]) -> str:
    if not missing_requirements:
        return ""

    questions = [item.user_question for item in missing_requirements]

    if len(questions) == 1:
        return questions[0]

    return (
        "I can plan this trip, but I need a few details first:\n"
        + "\n".join(f"- {question}" for question in questions)
    )


def _missing_field_reason(field_name: str) -> str:
    reasons = {
        "origin_airport": "Needed to search flights from the correct departure airport.",
        "origin_country": "Needed to check visa requirements.",
        "destination_city": "Needed to search flights, hotels, and activities.",
        "duration_days": "Needed to build a time-based plan and calculate hotel costs.",
        "total_budget": "Needed to keep the trip plan within budget.",
    }

    return reasons.get(field_name, "Required for full trip planning.")


def _missing_field_question(field_name: str) -> str:
    questions = {
        "origin_airport": (
            "Which airport are you flying from? Please use a 3-letter airport code, "
            "such as TLV, JFK, or LHR."
        ),
        "origin_country": (
            "What is your passport/origin country for visa requirements?"
        ),
        "destination_city": (
            "Which supported city should I plan for: Paris, London, Tokyo, New York, or Berlin?"
        ),
        "duration_days": "How many days should the trip be?",
        "total_budget": "What total budget should I use for the trip, in USD?",
    }

    return questions.get(field_name, f"Please provide: {field_name}.")
