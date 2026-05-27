"""
Planner agent — master trip planner for the cache-miss path.

Answers: "run everything and produce the final plan."

Architecture:
cache_miss
→ run_master_planner(...)
→ deterministic context extraction
→ optional replanning merge
→ async SLM context enrichment
→ async sub-agent execution
→ merge enriched context
→ dependency check
→ run newly-ready tasks
→ HITL question if critical info is still missing
→ final plan generation
"""

import asyncio
import json
from typing import Dict, List, Optional, Set

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.base import get_model
from src.agents.context_enricher import (
    enrich_trip_context_async,
    extract_trip_context_deterministic,
    merge_modified_trip_context,
    merge_trip_context,
)
from src.agents.planner_dependencies import (
    build_planner_dependency_graph,
    check_planner_dependencies,
)
from src.agents.planner_scheduler import build_scheduler_result, completed_tasks_from
from src.agents.sub_agents.experience_agent import ExperienceAgent
from src.agents.sub_agents.replanning_agent import analyze_replanning
from src.agents.sub_agents.stay_agent import StayAgent
from src.agents.sub_agents.transport_agent import TransportAgent
from src.graph.state import AgentState
from src.models.context_enrichment import PreferenceUpdate
from src.models.planner import (
    DependencyCheckResult,
    PlannerStatus,
    PlannerTaskType,
)
from src.models.trip_context import TripContext
from src.prompts.loader import get_prompt
from src.services.planner_result_parser import (
    build_structured_tool_results,
    extract_lowest_price_from_json,
)
from src.tools.calc_tools import calculate_trip_cost
from src.utils.logger import get_logger

logger = get_logger("planner")


def run_master_planner(state: AgentState) -> dict:
    """
    Synchronous wrapper for the async master planner.

    The current CLI uses graph.stream(...), so this wrapper keeps the graph node
    synchronous while the planner internally runs async tasks with asyncio.
    """
    return asyncio.run(_run_master_planner_async(state))


async def _run_master_planner_async(state: AgentState) -> dict:
    """
    Runs the master planner after semantic cache miss, HITL resume, or replanning.
    """
    is_hitl_resume = bool(
        state.get("pending_trip_context")
        or state.get("awaiting_user_clarification")
    )

    planning_mode = "replanning" if state.get("force_replan") else "full_planning"
    allowed_tasks: Optional[Set[str]] = None

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

    if state.get("force_replan") and state.get("trip_context"):
        old_context = TripContext(**state["trip_context"])
        modified_context = deterministic_context

        deterministic_context = merge_modified_trip_context(
            old_context=old_context,
            modified_context=modified_context,
        )

        replanning_result = analyze_replanning(
            old_context=old_context,
            new_context=deterministic_context,
            existing_results=state.get("planner_task_results", {}) or {},
        )

        existing_task_results = replanning_result.preserved_results
        allowed_tasks = {task.value for task in replanning_result.changed_tasks}

        total_results = (
            len(replanning_result.preserved_results)
            + len(replanning_result.invalidated_results)
        )
        reuse_ratio = (
            len(replanning_result.preserved_results) / total_results
            if total_results > 0
            else 1.0
        )

        logger.info(
            "Planner entered replanning mode. changed_tasks=%s preserved_results=%s",
            [task.value for task in replanning_result.changed_tasks],
            list(existing_task_results.keys()),
        )
        logger.info(
            "Planner preserved existing results during replanning: %s",
            list(replanning_result.preserved_results.keys()),
        )
        logger.info(
            "Planner invalidated results during replanning: %s",
            list(replanning_result.invalidated_results.keys()),
        )
        logger.info("Planner reuse ratio during replanning: %.2f", reuse_ratio)

    elif state.get("force_replan"):
        logger.info(
            "force_replan=True but no previous trip_context was found; running full planning flow."
        )

    initial_task_results = await run_sub_agents_async(
        context=deterministic_context,
        existing_results=existing_task_results,
        allowed_tasks=allowed_tasks,
    )

    enrichment_result = await enrichment_task
    merged_context = merge_trip_context(deterministic_context, enrichment_result)

    final_dependency_result = check_planner_dependencies(merged_context)

    combined_existing_results = {
        **existing_task_results,
        **initial_task_results,
    }

    logger.info(
        "Planner existing reusable results=%s",
        list(combined_existing_results.keys()),
    )

    task_results = await run_sub_agents_async(
        context=merged_context,
        existing_results=combined_existing_results,
        allowed_tasks=allowed_tasks,
    )

    completed_tasks = completed_tasks_from(task_results)

    dependency_graph = build_planner_dependency_graph(
        context=merged_context,
        completed_tasks=completed_tasks,
    )

    scheduler_result = build_scheduler_result(dependency_graph)
    _log_dependency_state(dependency_graph, scheduler_result)

    updates = _build_preference_state_updates(
        state,
        enrichment_result.preference_updates,
    )

    structured_results = build_structured_tool_results(
        context=merged_context,
        raw_results=task_results,
    )

    updates["trip_context"] = merged_context.model_dump()
    updates["context_enrichment_status"] = (
        "completed" if enrichment_result.confidence > 0 else "failed"
    )
    updates["planner_status"] = final_dependency_result.status.value
    updates["planning_mode"] = planning_mode
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

    completed_tasks = completed_tasks_from(task_results)

    dependency_graph = build_planner_dependency_graph(
        context=merged_context,
        completed_tasks=completed_tasks,
    )

    scheduler_result = build_scheduler_result(dependency_graph)
    _log_dependency_state(dependency_graph, scheduler_result)

    structured_results = build_structured_tool_results(
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
        planning_mode=planning_mode,
    )

    updates["planner_status"] = PlannerStatus.READY.value
    updates["messages"] = [AIMessage(content=final_answer)]
    updates["tool_call_count"] = state.get("tool_call_count", 0) + len(task_results)

    updates["awaiting_user_clarification"] = False
    updates["pending_trip_context"] = {}
    updates["pending_missing_fields"] = []
    updates["pending_hitl_question"] = ""
    updates["pending_planner_task_results"] = {}
    updates["force_replan"] = False
    
    logger.info("Master planner completed final plan. planning_mode=%s", planning_mode)
    return updates


async def run_sub_agents_async(
    context: TripContext,
    existing_results: Optional[Dict[str, str]] = None,
    allowed_tasks: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """
    Runs planner sub-agents in parallel and safely merges their independent results.

    Agents whose result keys are all already present in existing_results are skipped.
    When allowed_tasks is provided, only agents that can produce one of those tasks run.
    """
    covered = set(existing_results or {})

    agents = [
        agent
        for agent in [TransportAgent(), StayAgent(), ExperienceAgent()]
        if not all(key in covered for key in agent.result_keys)
        and (
            allowed_tasks is None
            or any(key in allowed_tasks for key in agent.result_keys)
        )
    ]

    logger.info(
        "Planner selected sub-agents: %s",
        [
            getattr(agent, "agent_name", agent.__class__.__name__)
            for agent in agents
        ],
    )

    logger.info(
        "Planner sub-agent selection context. covered=%s allowed_tasks=%s",
        sorted(covered),
        sorted(allowed_tasks) if allowed_tasks is not None else None,
    )

    merged_raw_results: Dict[str, str] = {**(existing_results or {})}

    if not agents:
        logger.info(
            "Planner skipped all sub-agents — all required results already preserved."
        )
        return merged_raw_results

    logger.info(
        "Planner running sub-agents in parallel. count=%d",
        len(agents),
    )

    results = await asyncio.gather(
        *[agent.run(context=context) for agent in agents],
        return_exceptions=True,
    )

    for agent, result in zip(agents, results):
        if isinstance(result, Exception):
            logger.error(
                "Sub-agent failed. agent=%s error_type=%s error=%s",
                getattr(agent, "agent_name", agent.__class__.__name__),
                type(result).__name__,
                result,
            )
            continue

        logger.info(
            "Sub-agent completed. agent=%s result_keys=%s",
            getattr(agent, "agent_name", agent.__class__.__name__),
            list(result.raw_results.keys()),
        )

        merged_raw_results.update(result.raw_results)

    return merged_raw_results


async def _calculate_cost_if_possible(
    context: TripContext,
    task_results: Dict[str, str],
) -> Optional[str]:
    """
    Calculates total trip cost when flight, hotel, and duration are available.
    """
    if context.duration_days is None:
        return None

    flight_price = extract_lowest_price_from_json(
        task_results.get(PlannerTaskType.FETCH_FLIGHTS.value, "")
    )

    hotel_price = extract_lowest_price_from_json(
        task_results.get(PlannerTaskType.FETCH_HOTELS.value, ""),
        price_key="price_per_night",
    )

    if flight_price is None or hotel_price is None:
        logger.info(
            "Cost calculation skipped. flight_price=%s hotel_price=%s duration_days=%s",
            flight_price,
            hotel_price,
            context.duration_days,
        )
        return None

    logger.info(
        "Calculating trip cost. flight_price=%s hotel_price=%s duration_days=%s",
        flight_price,
        hotel_price,
        context.duration_days,
    )

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
    planning_mode: str = "full_planning",
) -> str:
    """
    Generates the final user-facing travel plan.
    """
    model = get_model(temperature=0)

    response = await model.ainvoke([
        SystemMessage(content=get_prompt("final_answer_prompt")),
        HumanMessage(
            content=(
                f"Planning mode: {planning_mode}\n\n"
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


def _log_dependency_state(dependency_graph, scheduler_result) -> None:
    """
    Logs dependency graph and scheduler status for observability.
    """
    logger.info(
        "Planner dependency graph ready=%s blocked=%s completed=%s",
        [task.value for task in dependency_graph.ready_tasks],
        [task.value for task in dependency_graph.blocked_tasks],
        [task.value for task in dependency_graph.completed_tasks],
    )

    logger.info(
        "Planner scheduler waves=%s",
        [
            {
                "wave": wave.wave_number,
                "tasks": [task.value for task in wave.tasks],
            }
            for wave in scheduler_result.waves
        ],
    )


def _build_default_hitl_question() -> str:
    return (
        "I can plan this trip, but I need the origin airport, origin country, "
        "destination city, trip duration, and total budget first."
    )