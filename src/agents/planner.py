from typing import Dict, List, Optional
"""
Planner agent — master trip planner for the cache-miss path.

This module contains:
1. Dependency analysis for trip planning.
2. Async execution of ready planner tasks.
3. Master planner orchestration after semantic cache miss.

Architecture:
cache_miss
→ run_master_planner(...)
→ deterministic context extraction
→ async SLM context enrichment
→ dependency check
→ async ready task execution
→ merge enriched context
→ run newly-ready tasks
→ HITL question if critical info is still missing
→ final plan generation
"""

import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.agents.sub_agents.transport_agent import TransportAgent
from src.agents.sub_agents.stay_agent import StayAgent
from src.agents.sub_agents.experience_agent import ExperienceAgent
from src.agents.base import get_model
from src.agents.context_enricher import (
    enrich_trip_context_async,
    extract_trip_context_deterministic,
    merge_trip_context,
)
from src.graph.state import AgentState
from src.models.context_enrichment import PreferenceUpdate
from src.models.planner import (
    DependencyCheckResult,
    MissingRequirement,
    PlannerStatus,
    PlannerTask,
    PlannerTaskStatus,
    PlannerTaskType,
    ActivityResult,
    CostResult,
    FlightResult,
    HotelResult,
    PlannerToolResults,
    VisaResult,
    DependencyStatus,
    PlannerDependency,
    PlannerDependencyGraph,
    PlannerTaskNode,
    SchedulerResult,
    SchedulerWave,
)
from src.models.trip_context import REQUIRED_TRIP_FIELDS, TripContext
from src.tools.calc_tools import calculate_trip_cost
from src.utils.logger import get_logger
from src.prompts.loader import get_prompt

logger = get_logger("planner")


_TASK_REQUIREMENTS: Dict[PlannerTaskType, tuple[str, ...]] = {
    PlannerTaskType.FETCH_FLIGHTS: ("origin_airport", "destination_city"),
    PlannerTaskType.FETCH_HOTELS: ("destination_city",),
    PlannerTaskType.FETCH_ACTIVITIES: ("destination_city",),
    PlannerTaskType.CHECK_VISA: ("origin_country", "destination_country"),

    # Cost calculation is special: it also depends on tool results
    # from fetch_flights and fetch_hotels. Those are checked later in
    # _calculate_cost_if_possible().
    PlannerTaskType.CALCULATE_TRIP_COST: ("duration_days",),
}

async def run_sub_agents_async(
    context: TripContext,
    existing_results: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Runs planner sub-agents in parallel and safely merges their independent results.
    """
    agents = [
        TransportAgent(),
        StayAgent(),
        ExperienceAgent(),
    ]

    results = await asyncio.gather(
        *[
            agent.run(context=context)
            for agent in agents
        ],
        return_exceptions=True,
    )

    merged_raw_results: Dict[str, str] = {
        **(existing_results or {})
    }

    for agent, result in zip(agents, results):
        if isinstance(result, Exception):
            logger.error(
                "Sub-agent failed. agent=%s error=%s",
                getattr(agent, "agent_name", agent.__class__.__name__),
                result,
            )
            continue

        merged_raw_results.update(result.raw_results)

    return merged_raw_results

def run_master_planner(state: AgentState) -> dict:
    """
    Synchronous wrapper for the async master planner.

    The current CLI uses graph.stream(...), so this wrapper keeps the graph node
    synchronous while the planner internally runs async tasks with asyncio.
    """
    return asyncio.run(_run_master_planner_async(state))

def build_planner_dependency_graph(
    context: TripContext,
    completed_tasks: Optional[List[PlannerTaskType]] = None,
) -> PlannerDependencyGraph:
    """
    Builds an explicit planner dependency DAG.

    This does not replace the current execution logic yet.
    It gives the planner a structured graph that future async scheduling can use.
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


async def _run_master_planner_async(state: AgentState) -> dict:
    """
    Runs the master planner after semantic cache miss or HITL resume.
    """
    is_hitl_resume = bool(
        state.get("pending_trip_context")
        or state.get("awaiting_user_clarification")
    )

    if is_hitl_resume and state.get("trip_context"):
        deterministic_context = TripContext(**state["trip_context"])
        logger.info("Master planner resumed from HITL trip_context.")

    elif is_hitl_resume and state.get("pending_trip_context"):
        deterministic_context = TripContext(**state["pending_trip_context"])
        logger.info("Master planner resumed from pending_trip_context.")

    else:
        deterministic_context = extract_trip_context_deterministic(state)
        logger.info("Master planner created fresh deterministic context.")

    enrichment_task = asyncio.create_task(
        enrich_trip_context_async(state, deterministic_context)
    )

    existing_task_results = (
        state.get("pending_planner_task_results", {}) or {}
        if is_hitl_resume
        else {}
    )

    initial_task_results = await run_sub_agents_async(
        context=deterministic_context,
        existing_results=existing_task_results,
    )

    enrichment_result = await enrichment_task
    merged_context = merge_trip_context(deterministic_context, enrichment_result)

    final_dependency_result = check_planner_dependencies(merged_context)

    task_results = await run_sub_agents_async(
        context=merged_context,
        existing_results={
            **existing_task_results,
            **initial_task_results,
        },
    )

    completed_tasks = [
        PlannerTaskType(task_type)
        for task_type in task_results.keys()
        if task_type in {item.value for item in PlannerTaskType}
    ]

    dependency_graph = build_planner_dependency_graph(
        context=merged_context,
        completed_tasks=completed_tasks,
    )

    scheduler_result = build_scheduler_result(dependency_graph)

    updates = _build_preference_state_updates(
        state,
        enrichment_result.preference_updates,
    )

    structured_results = _build_structured_tool_results(
        context=merged_context,
        raw_results=task_results,
    )

    updates["trip_context"] = merged_context.model_dump()
    updates["context_enrichment_status"] = (
        "completed" if enrichment_result.confidence > 0 else "failed"
    )
    updates["planner_status"] = final_dependency_result.status.value
    updates["planner_task_results"] = task_results
    updates["planner_structured_results"] = structured_results.model_dump()
    updates["planner_dependency_graph"] = dependency_graph.model_dump()
    updates["planner_scheduler_result"] = scheduler_result.model_dump()

    if final_dependency_result.missing_requirements:
        hitl_question = (
            final_dependency_result.hitl_question
            or _build_default_hitl_question()
        )

        updates["planner_status"] = PlannerStatus.MISSING_REQUIRED_INFO.value
        updates["awaiting_user_clarification"] = True
        updates["pending_trip_context"] = merged_context.model_dump()
        updates["pending_missing_fields"] = [
            item.field_name
            for item in final_dependency_result.missing_requirements
        ]
        updates["pending_hitl_question"] = hitl_question
        updates["pending_planner_task_results"] = task_results
        updates["messages"] = [AIMessage(content=hitl_question)]

        logger.info("Master planner stopped for HITL: %s", hitl_question)

        return updates

    cost_result = await _calculate_cost_if_possible(
        context=merged_context,
        task_results=task_results,
    )

    if cost_result:
        task_results[PlannerTaskType.CALCULATE_TRIP_COST.value] = cost_result
        updates["planner_task_results"] = task_results

    completed_tasks = [
        PlannerTaskType(task_type)
        for task_type in task_results.keys()
        if task_type in {item.value for item in PlannerTaskType}
    ]

    dependency_graph = build_planner_dependency_graph(
        context=merged_context,
        completed_tasks=completed_tasks,
    )

    scheduler_result = build_scheduler_result(dependency_graph)

    structured_results = _build_structured_tool_results(
        context=merged_context,
        raw_results=task_results,
    )

    updates["planner_structured_results"] = structured_results.model_dump()
    updates["planner_dependency_graph"] = dependency_graph.model_dump()
    updates["planner_scheduler_result"] = scheduler_result.model_dump()

    final_answer = await _generate_final_plan(
        context=merged_context,
        dependency_result=final_dependency_result,
        task_results=task_results,
    )

    updates["planner_status"] = PlannerStatus.READY.value
    updates["messages"] = [AIMessage(content=final_answer)]
    updates["tool_call_count"] = state.get("tool_call_count", 0) + len(task_results)

    updates["awaiting_user_clarification"] = False
    updates["pending_trip_context"] = {}
    updates["pending_missing_fields"] = []
    updates["pending_hitl_question"] = ""
    updates["pending_planner_task_results"] = {}

    logger.info("Master planner completed final plan.")
    return updates


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
                reason="Tasks with all required context and completed dependencies.",
            )
        )

    return SchedulerResult(
        waves=waves,
        completed_tasks=dependency_graph.completed_tasks,
        blocked_tasks=dependency_graph.blocked_tasks,
        reason=(
            "Scheduler built from current dependency graph. "
            "Ready tasks can run concurrently in the first wave."
        ),
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
    """
    Builds missing critical field objects for full trip planning.
    """
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
    """
    Builds a PlannerTask based on task-specific required fields.
    """
    required_fields = _TASK_REQUIREMENTS[task_type]
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

async def _calculate_cost_if_possible(
    context: TripContext,
    task_results: Dict[str, str],
) -> Optional[str]:
    """
    Calculates trip cost if flight and hotel data are available.
    """
    if context.duration_days is None:
        return None

    flight_price = _extract_lowest_price_from_json(
        task_results.get(PlannerTaskType.FETCH_FLIGHTS.value, "")
    )

    hotel_price = _extract_lowest_price_from_json(
        task_results.get(PlannerTaskType.FETCH_HOTELS.value, ""),
        price_key="price_per_night",
    )

    if flight_price is None or hotel_price is None:
        return None

    return await asyncio.to_thread(
        calculate_trip_cost.invoke,
        {
            "flight_price": flight_price,
            "hotel_price_per_night": hotel_price,
            "duration_days": context.duration_days,
        },
    )


async def _generate_final_plan(
    context: TripContext,
    dependency_result: DependencyCheckResult,
    task_results: Dict[str, str],
) -> str:
    """
    Uses the LLM to generate a final user-facing travel plan from structured data.
    """
    model = get_model(temperature=0)

    response = await model.ainvoke([
        SystemMessage(content=get_prompt("final_answer_prompt")),
        HumanMessage(
            content=(
                "TripContext:\n"
                f"{context.model_dump()}\n\n"
                "DependencyCheckResult:\n"
                f"{dependency_result.model_dump()}\n\n"
                "Tool results:\n"
                f"{json.dumps(task_results, indent=2)}"
            )
        ),
    ])

    content = response.content
    return content if isinstance(content, str) else str(content)


def _build_preference_state_updates(
    state: AgentState,
    preference_updates: List[PreferenceUpdate],
) -> dict:
    """
    Converts persistent preference updates into AgentState field updates.
    """
    updates: dict = {}

    for update in preference_updates:
        if update.field_name == "travel_preferences":
            existing = state.get("travel_preferences", "") or ""
            pending = updates.get("travel_preferences", existing) or ""

            updates["travel_preferences"] = (
                f"{pending}\n- {update.value}".strip()
                if pending
                else f"- {update.value}"
            )
            continue

        updates[update.field_name] = update.value

    return updates


def _extract_lowest_price_from_json(
    raw_json: str,
    *,
    price_key: str = "price",
) -> Optional[float]:
    """
    Extracts the lowest price from a tool JSON response.
    """
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(data, dict):
        value = data.get(price_key)
        return float(value) if value is not None else None

    if not isinstance(data, list):
        return None

    prices = []

    for item in data:
        if not isinstance(item, dict):
            continue

        value = item.get(price_key)
        if value is None:
            continue

        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            continue

    return min(prices) if prices else None


def _build_hitl_question(missing_requirements: List[MissingRequirement]) -> str:
    """
    Builds a concise HITL question from missing critical fields.
    """
    if not missing_requirements:
        return ""

    questions = [item.user_question for item in missing_requirements]

    if len(questions) == 1:
        return questions[0]

    return (
        "I can plan this trip, but I need a few details first:\n"
        + "\n".join(f"- {question}" for question in questions)
    )


def _build_default_hitl_question() -> str:
    """
    Fallback HITL question.
    """
    return (
        "I can plan this trip, but I need the origin airport, origin country, "
        "destination city, trip duration, and total budget first."
    )


def _missing_field_reason(field_name: str) -> str:
    """
    Explains why a critical field is required.
    """
    reasons = {
        "origin_airport": "Needed to search flights from the correct departure airport.",
        "origin_country": "Needed to check visa requirements.",
        "destination_city": "Needed to search flights, hotels, and activities.",
        "duration_days": "Needed to build a time-based plan and calculate hotel costs.",
        "total_budget": "Needed to keep the trip plan within budget.",
    }

    return reasons.get(field_name, "Required for full trip planning.")


def _build_structured_tool_results(
    *,
    context: TripContext,
    raw_results: Dict[str, str],
) -> PlannerToolResults:
    """
    Builds a structured planner output container from existing raw tool results.

    This keeps backward compatibility:
    - planner_task_results remains raw dict[str, str]
    - planner_structured_results becomes typed/shared data for future agents
    """
    return PlannerToolResults(
        flights=_parse_flight_results(
            raw_results.get(PlannerTaskType.FETCH_FLIGHTS.value, "")
        ),
        hotels=_parse_hotel_results(
            raw_results.get(PlannerTaskType.FETCH_HOTELS.value, "")
        ),
        activities=_parse_activity_results(
            raw_results.get(PlannerTaskType.FETCH_ACTIVITIES.value, "")
        ),
        visa=_parse_visa_result(
            raw_results.get(PlannerTaskType.CHECK_VISA.value, ""),
            context=context,
        ),
        cost=_parse_cost_result(
            raw_results.get(PlannerTaskType.CALCULATE_TRIP_COST.value, ""),
            context=context,
        ),
        raw_results=raw_results,
    )


def _parse_json_result(raw: str):
    """
    Safely parses a JSON-like tool result.

    Returns None when the tool output is not valid JSON.
    """
    if not raw:
        return None

    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _ensure_list(data) -> list:
    """
    Normalizes parsed tool data into a list.
    """
    if data is None:
        return []

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data]

    return []


def _as_float(value) -> Optional[float]:
    """
    Converts a value to float when possible.
    """
    if value in (None, ""):
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    """
    Converts a value to int when possible.
    """
    if value in (None, ""):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_flight_results(raw: str) -> List[FlightResult]:
    """
    Parses raw flight tool output into structured FlightResult objects.
    """
    rows = _ensure_list(_parse_json_result(raw))
    results: List[FlightResult] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        results.append(
            FlightResult(
                origin=row.get("origin"),
                destination=row.get("destination"),
                airline=row.get("airline"),
                flight_number=row.get("flight_number") or row.get("flight"),
                price=_as_float(row.get("price")),
                departure_time=row.get("departure_time"),
                arrival_time=row.get("arrival_time"),
                duration=row.get("duration"),
                raw=row,
            )
        )

    return results


def _parse_hotel_results(raw: str) -> List[HotelResult]:
    """
    Parses raw hotel tool output into structured HotelResult objects.
    """
    rows = _ensure_list(_parse_json_result(raw))
    results: List[HotelResult] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        results.append(
            HotelResult(
                city=row.get("city"),
                name=row.get("name") or row.get("hotel"),
                price_per_night=_as_float(row.get("price_per_night")),
                rating=_as_float(row.get("rating")),
                location=row.get("location"),
                raw=row,
            )
        )

    return results


def _parse_activity_results(raw: str) -> List[ActivityResult]:
    """
    Parses raw activity tool output into structured ActivityResult objects.
    """
    rows = _ensure_list(_parse_json_result(raw))
    results: List[ActivityResult] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        results.append(
            ActivityResult(
                city=row.get("city"),
                name=row.get("name") or row.get("activity"),
                category=row.get("category"),
                price=_as_float(row.get("price")),
                duration=row.get("duration"),
                raw=row,
            )
        )

    return results


def _parse_visa_result(
    raw: str,
    *,
    context: TripContext,
) -> Optional[VisaResult]:
    """
    Parses raw visa tool output into a structured VisaResult.
    """
    data = _parse_json_result(raw)

    if not isinstance(data, dict):
        if not raw:
            return None

        return VisaResult(
            origin_country=context.origin_country,
            destination_country=context.destination_country,
            requirement_summary=raw,
            raw={},
        )

    return VisaResult(
        origin_country=data.get("origin_country") or context.origin_country,
        destination_country=(
            data.get("destination_country") or context.destination_country
        ),
        visa_required=data.get("visa_required"),
        requirement_summary=(
            data.get("requirement_summary")
            or data.get("requirement")
            or data.get("summary")
        ),
        raw=data,
    )


def _parse_cost_result(
    raw: str,
    *,
    context: TripContext,
) -> Optional[CostResult]:
    """
    Parses raw cost tool output into a structured CostResult.
    """
    data = _parse_json_result(raw)

    if not isinstance(data, dict):
        return None

    total_cost = (
        _as_float(data.get("total_cost"))
        or _as_float(data.get("total"))
        or _as_float(data.get("estimated_total"))
    )

    within_budget = None
    if total_cost is not None and context.total_budget is not None:
        within_budget = total_cost <= context.total_budget

    return CostResult(
        flight_price=_as_float(data.get("flight_price")),
        hotel_price_per_night=_as_float(data.get("hotel_price_per_night")),
        duration_days=_as_int(data.get("duration_days")) or context.duration_days,
        total_cost=total_cost,
        within_budget=within_budget,
        raw=data,
    )


def _missing_field_question(field_name: str) -> str:
    """
    Builds a user-facing question for a missing critical field.
    """
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