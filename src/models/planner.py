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

class FlightResult(BaseModel):
    """
    Structured result for a single flight option.

    This model is intentionally permissive so it can wrap both demo SQLite data
    and future external API data without breaking the planner.
    """

    origin: str | None = Field(
        default=None,
        description="Departure airport or city."
    )

    destination: str | None = Field(
        default=None,
        description="Destination airport or city."
    )

    airline: str | None = Field(
        default=None,
        description="Airline name or code."
    )

    flight_number: str | None = Field(
        default=None,
        description="Flight number, if available."
    )

    price: float | None = Field(
        default=None,
        description="Flight price in USD, if available."
    )

    departure_time: str | None = Field(
        default=None,
        description="Departure time, if available."
    )

    arrival_time: str | None = Field(
        default=None,
        description="Arrival time, if available."
    )

    duration: str | None = Field(
        default=None,
        description="Flight duration, if available."
    )

    raw: dict = Field(
        default_factory=dict,
        description="Original raw provider/tool payload."
    )


class HotelResult(BaseModel):
    """
    Structured result for a single hotel option.
    """

    city: str | None = Field(
        default=None,
        description="Hotel city."
    )

    name: str | None = Field(
        default=None,
        description="Hotel name."
    )

    price_per_night: float | None = Field(
        default=None,
        description="Hotel price per night in USD, if available."
    )

    rating: float | None = Field(
        default=None,
        description="Hotel rating, if available."
    )

    location: str | None = Field(
        default=None,
        description="Hotel location or neighborhood."
    )

    raw: dict = Field(
        default_factory=dict,
        description="Original raw provider/tool payload."
    )


class ActivityResult(BaseModel):
    """
    Structured result for a single activity option.
    """

    city: str | None = Field(
        default=None,
        description="Activity city."
    )

    name: str | None = Field(
        default=None,
        description="Activity name."
    )

    category: str | None = Field(
        default=None,
        description="Activity category."
    )

    price: float | None = Field(
        default=None,
        description="Activity price in USD, if available."
    )

    duration: str | None = Field(
        default=None,
        description="Activity duration, if available."
    )

    raw: dict = Field(
        default_factory=dict,
        description="Original raw provider/tool payload."
    )


class VisaResult(BaseModel):
    """
    Structured visa requirement result.
    """

    origin_country: str | None = Field(
        default=None,
        description="Traveler origin/passport country."
    )

    destination_country: str | None = Field(
        default=None,
        description="Destination country."
    )

    visa_required: bool | None = Field(
        default=None,
        description="Whether a visa is required."
    )

    requirement_summary: str | None = Field(
        default=None,
        description="Short visa requirement summary."
    )

    raw: dict = Field(
        default_factory=dict,
        description="Original raw provider/tool payload."
    )


class CostResult(BaseModel):
    """
    Structured cost calculation result.
    """

    flight_price: float | None = Field(
        default=None,
        description="Selected or lowest flight price in USD."
    )

    hotel_price_per_night: float | None = Field(
        default=None,
        description="Selected or lowest hotel price per night in USD."
    )

    duration_days: int | None = Field(
        default=None,
        description="Trip duration in days."
    )

    total_cost: float | None = Field(
        default=None,
        description="Estimated total trip cost in USD."
    )

    within_budget: bool | None = Field(
        default=None,
        description="Whether the estimated total cost is within the user's budget."
    )

    raw: dict = Field(
        default_factory=dict,
        description="Original raw provider/tool payload."
    )


class PlannerToolResults(BaseModel):
    """
    Shared structured output container for planner tools and future sub-agents.

    The goal is to let future agents consume each other's results without
    parsing unstructured strings.
    """

    flights: list[FlightResult] = Field(
        default_factory=list,
        description="Structured flight results."
    )

    hotels: list[HotelResult] = Field(
        default_factory=list,
        description="Structured hotel results."
    )

    activities: list[ActivityResult] = Field(
        default_factory=list,
        description="Structured activity results."
    )

    visa: VisaResult | None = Field(
        default=None,
        description="Structured visa result."
    )

    cost: CostResult | None = Field(
        default=None,
        description="Structured cost result."
    )

    raw_results: dict[str, str] = Field(
        default_factory=dict,
        description="Backward-compatible raw string results keyed by PlannerTaskType value."
    )


class DependencyStatus(str, Enum):
    """
    Status of a task inside the explicit planner dependency graph.
    """

    READY = "ready"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class PlannerDependency(BaseModel):
    """
    A dependency edge between planner tasks.

    Example:
      calculate_trip_cost depends on fetch_flights and fetch_hotels.
    """

    task: PlannerTaskType = Field(
        description="The task that depends on another task."
    )

    depends_on: PlannerTaskType = Field(
        description="The task that must complete first."
    )

    reason: str = Field(
        description="Why this dependency exists."
    )


class PlannerTaskNode(BaseModel):
    """
    A task node inside the dependency graph.
    """

    task_type: PlannerTaskType = Field(
        description="Planner task represented by this node."
    )

    status: DependencyStatus = Field(
        default=DependencyStatus.BLOCKED,
        description="Current dependency-graph status for this task."
    )

    required_context_fields: list[str] = Field(
        default_factory=list,
        description="TripContext fields required before this task can run."
    )

    depends_on: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Other planner tasks that must finish before this task can run."
    )

    unlocked_by: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks that become closer to ready when this task completes."
    )

    reason: str = Field(
        default="",
        description="Explanation of the current node status."
    )

class PlannerDependencyGraph(BaseModel):
    """
    Explicit dependency graph for planner execution.

    This is the foundation for the async dependency scheduler.
    """

    nodes: dict[PlannerTaskType, PlannerTaskNode] = Field(
        default_factory=dict,
        description="Planner task nodes keyed by task type."
    )

    dependencies: list[PlannerDependency] = Field(
        default_factory=list,
        description="Explicit dependency edges between planner tasks."
    )

    ready_tasks: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks currently ready to execute."
    )

    blocked_tasks: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks currently blocked by missing context or dependencies."
    )

    completed_tasks: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks already completed."
    )

class SchedulerWave(BaseModel):
    """
    A single async execution wave.

    Tasks inside the same wave can run in parallel.
    """

    wave_number: int = Field(
        description="Execution wave number, starting from 1."
    )

    tasks: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks scheduled to run in this wave."
    )

    reason: str = Field(
        default="",
        description="Why these tasks are grouped in this wave."
    )


class SchedulerResult(BaseModel):
    """
    Result of building an async dependency-aware execution schedule.
    """

    waves: list[SchedulerWave] = Field(
        default_factory=list,
        description="Ordered execution waves."
    )

    completed_tasks: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks considered completed before or during scheduling."
    )

    blocked_tasks: list[PlannerTaskType] = Field(
        default_factory=list,
        description="Tasks that could not be scheduled."
    )

    reason: str = Field(
        default="",
        description="High-level scheduler explanation."
    )