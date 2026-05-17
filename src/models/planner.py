from enum import Enum
from pydantic import BaseModel, Field


class PlannerStatus(str, Enum):
    """
    High-level planning status after dependency analysis.
    """

    READY = "ready"
    PARTIAL_READY = "partial_ready"
    MISSING_REQUIRED_INFO = "missing_required_info"


class PlannerTaskType(str, Enum):
    """
    Tool/sub-agent tasks the master planner can schedule.
    """

    FETCH_FLIGHTS = "fetch_flights"
    FETCH_HOTELS = "fetch_hotels"
    FETCH_ACTIVITIES = "fetch_activities"
    CHECK_VISA = "check_visa"
    CALCULATE_TRIP_COST = "calculate_trip_cost"


class PlannerTaskStatus(str, Enum):
    """
    Execution readiness status for a planner task.
    """

    READY = "ready"
    BLOCKED = "blocked"
    WAITING_FOR_ENRICHMENT = "waiting_for_enrichment"


class MissingRequirement(BaseModel):
    """
    A required field missing from the trip context.
    """

    field_name: str = Field(
        description="The missing TripContext field name."
    )

    reason: str = Field(
        description="Why this field is needed before full planning can continue."
    )

    user_question: str = Field(
        description="A concise question to ask the user for this missing field."
    )


class PlannerTask(BaseModel):
    """
    A planner task and whether it can run with the currently available context.
    """

    task_type: PlannerTaskType = Field(
        description="The task type to run."
    )

    status: PlannerTaskStatus = Field(
        description="Whether this task is ready, blocked, or waiting for enrichment."
    )

    is_ready: bool = Field(
        description="Whether this task has all required inputs and can run immediately."
    )

    missing_fields: list[str] = Field(
        default_factory=list,
        description="Fields still required before this task can run."
    )

    can_run_async: bool = Field(
        default=False,
        description="Whether this task can be executed asynchronously without blocking the planner."
    )

    reason: str = Field(
        description="Why this task is ready or blocked."
    )


class DependencyCheckResult(BaseModel):
    """
    Result of checking TripContext against required fields and task dependencies.
    """

    status: PlannerStatus = Field(
        description="Overall planner readiness status."
    )

    missing_requirements: list[MissingRequirement] = Field(
        default_factory=list,
        description="Critical missing fields required for a full trip plan."
    )

    ready_tasks: list[PlannerTask] = Field(
        default_factory=list,
        description="Tasks that can run immediately."
    )

    blocked_tasks: list[PlannerTask] = Field(
        default_factory=list,
        description="Tasks that cannot run yet because required fields are missing."
    )

    async_ready_tasks: list[PlannerTask] = Field(
        default_factory=list,
        description="Tasks that can be started asynchronously right now."
    )

    hitl_question: str | None = Field(
        default=None,
        description="Human-in-the-loop question to ask when critical information is missing."
    )