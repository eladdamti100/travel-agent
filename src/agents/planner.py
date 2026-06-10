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
import concurrent.futures
from typing import Dict, List, Optional, Set

from langchain_core.messages import AIMessage

from src.agents.context_enricher import (
    enrich_trip_context_async,
    extract_trip_context_deterministic,
    extract_trip_context_from_history,
    merge_modified_trip_context,
    merge_trip_context,
)
from src.agents.hitl_feedback_parser import apply_hitl_feedback
from src.agents.planner_dependencies import (
    build_planner_dependency_graph,
    check_planner_dependencies,
)
from src.agents.planner_scheduler import build_scheduler_result, completed_tasks_from
from src.agents.sub_agents.replanning_agent import analyze_replanning
from src.agents.db_supervisor import DBSupervisor
from src.agents.web_supervisor import WebSupervisor
from src.config.city_registry import COUNTRY_BY_CITY as _DESTINATION_COUNTRY_BY_CITY
from src.config.settings import settings
from src.graph.state import AgentState
from src.models.context_enrichment import ContextEnrichmentResult, PreferenceUpdate
from src.models.planner import PlannerStatus, PlannerTaskType
from src.models.trip_context import TripContext
from src.services.plan_enricher import calculate_cost_if_possible, fill_missing_with_web
from src.services.plan_generator import generate_final_plan
from src.services.planner_result_parser import build_structured_tool_results
from src.utils.logger import get_logger

logger = get_logger("planner")


def run_master_planner(state: AgentState) -> dict:
    """
    Synchronous wrapper for the async master planner.

    Uses a dedicated thread so the coroutine always gets a clean event loop,
    regardless of whether the caller is already inside a running loop (FastAPI,
    LangGraph server mode, Jupyter, pytest-asyncio).  asyncio.run() would raise
    RuntimeError: "This event loop is already running" in those contexts.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, _run_master_planner_async(state))
        return future.result()


async def _run_master_planner_async(state: AgentState) -> dict:
    """
    Runs the master planner after semantic cache miss, HITL resume, or replanning.
    """
    is_hitl_resume = bool(
        state.get("pending_trip_context")
        or state.get("awaiting_user_clarification")
    )

    hitl_feedback: str = state.get("hitl_feedback") or ""

    # Check if a previous trip context actually exists in the state
    has_previous_context = bool(state.get("trip_context"))

    # Set replanning ONLY if forced AND we actually have something to re-plan
    planning_mode = "replanning" if (state.get("force_replan") and has_previous_context) else "full_planning"
    if is_hitl_resume and state.get("trip_context"):
        deterministic_context = TripContext(**state["trip_context"])
        logger.info("Master planner resumed from HITL trip_context.")

    elif is_hitl_resume and state.get("pending_trip_context"):
        deterministic_context = TripContext(**state["pending_trip_context"])
        logger.info("Master planner resumed from pending_trip_context.")

    elif state.get("trip_context") and (state.get("critic_attempts") or 0) > 0:
        # Critic auto-replan: keep the previously confirmed trip context so the
        # planner does not re-ask for fields that were already provided.
        deterministic_context = TripContext(**state["trip_context"])
        logger.info("Master planner reusing existing trip_context for critic replan.")

    else:
        deterministic_context = extract_trip_context_deterministic(state)
        logger.info("Master planner created fresh deterministic context.")

    # Apply HITL edit overrides (origin, destination, duration, budget).
    if state.get("force_replan") and hitl_feedback:
        deterministic_context = apply_hitl_feedback(deterministic_context, hitl_feedback)

    allowed_tasks: Optional[Set[str]] = None

    is_critic_replan = (state.get("critic_attempts") or 0) > 0 and not is_hitl_resume

    critic_issues: list = []
    critic_suggestions: list = []
    if is_critic_replan:
        critique = state.get("critique_result") or {}
        critic_issues = critique.get("issues") or []
        critic_suggestions = critique.get("suggestions") or []
        logger.info(
            "Planner auto-replan triggered by critic. issues=%s suggestions=%s",
            critic_issues,
            critic_suggestions,
        )

    existing_task_results = (
        state.get("pending_planner_task_results", {}) or {}
        if is_hitl_resume
        else state.get("planner_task_results", {}) or {}
        if is_critic_replan
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
            hitl_feedback=hitl_feedback,
        )

        existing_task_results = replanning_result.preserved_results
        # changed_tasks is List[str] from diff_changed_tasks — no .value needed
        allowed_tasks = set(replanning_result.changed_tasks)

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
            replanning_result.changed_tasks,
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
        # No persisted trip_context (e.g. checkpoint didn't survive a HITL
        # interrupt). Recover prior trip details from earlier user messages
        # so a follow-up edit like "now my budget is 600 dollars" doesn't
        # re-ask for fields the user already provided.
        old_context = extract_trip_context_from_history(state)
        has_history_context = any(
            value not in (None, "", [])
            for field_name, value in old_context.model_dump().items()
            if field_name not in ("extraction_source", "slm_enriched")
        )
        if has_history_context:
            deterministic_context = merge_modified_trip_context(
                old_context=old_context,
                modified_context=deterministic_context,
            )
            planning_mode = "replanning"
            logger.info(
                "force_replan=True with no persisted trip_context; "
                "recovered context from message history."
            )
        else:
            logger.info(
                "force_replan=True but no previous trip_context was found; running full planning flow."
            )

    # P1-4.3 / P1-2.3: Start enrichment AFTER replanning (so it sees the final
    # deterministic_context) and run it concurrently with the first sub-agent wave.
    # asyncio.wait_for gives a hard timeout so a slow LLM never stalls the planner.
    _enrich_coro = asyncio.wait_for(
        enrich_trip_context_async(state, deterministic_context),
        timeout=settings.enrichment_timeout_seconds,
    )
    _subagent_coro = run_sub_agents_async(
        context=deterministic_context,
        existing_results=existing_task_results,
        allowed_tasks=allowed_tasks,
    )
    _enrichment_raw, initial_task_results = await asyncio.gather(
        _enrich_coro, _subagent_coro, return_exceptions=True
    )

    if isinstance(_enrichment_raw, Exception):
        logger.warning(
            "Enrichment failed or timed out (%s). Falling back to deterministic context.",
            type(_enrichment_raw).__name__,
        )
        enrichment_result = ContextEnrichmentResult(trip_context=deterministic_context)
    else:
        enrichment_result = _enrichment_raw

    if isinstance(initial_task_results, Exception):
        logger.error("Initial sub-agent wave failed: %s", initial_task_results)
        initial_task_results = {}

    merged_context = merge_trip_context(deterministic_context, enrichment_result)

    # Early-exit: destination not in supported list — no DB data exists for it.
    _SUPPORTED_DESTINATIONS = set(_DESTINATION_COUNTRY_BY_CITY.keys())
    if merged_context.destination_city and merged_context.destination_city not in _SUPPORTED_DESTINATIONS:
        _unsupported = merged_context.destination_city
        _supported_list = ", ".join(sorted(_SUPPORTED_DESTINATIONS))
        _msg = (
            f"Sorry, **{_unsupported}** is not yet a supported destination.\n\n"
            f"Please choose one of the available cities: {_supported_list}."
        )
        logger.info("Planner early-exit: unsupported destination=%s", _unsupported)
        return {
            "planner_status": PlannerStatus.MISSING_REQUIRED_INFO.value,
            "awaiting_user_clarification": False,
            "messages": [AIMessage(content=_msg)],
            "trip_context": merged_context.model_dump(),
            "planning_mode": planning_mode,
        }

    final_dependency_result = check_planner_dependencies(merged_context)

    # Second sub-agent run: uses merged_context (enrichment may have unlocked
    # new tasks e.g. fetch_weather via travel_month); skips already-covered keys.
    combined_existing_results = {**existing_task_results, **initial_task_results}
    logger.info("Planner reusable results after first wave=%s", list(combined_existing_results.keys()))

    task_results = await run_sub_agents_async(
        context=merged_context,
        existing_results=combined_existing_results,
        allowed_tasks=allowed_tasks,
    )

    updates = _build_preference_state_updates(state, enrichment_result.preference_updates)
    updates["trip_context"] = merged_context.model_dump()
    updates["context_enrichment_status"] = (
        "completed" if enrichment_result.confidence > 0 else "failed"
    )
    updates["planning_mode"] = planning_mode

    if final_dependency_result.missing_requirements:
        # P1-4.4: compute dependency state only for the HITL return path —
        # on the happy path a single computation happens below after web fallback.
        _completed = completed_tasks_from(task_results)
        _dep_graph = build_planner_dependency_graph(context=merged_context, completed_tasks=_completed)
        _sched = build_scheduler_result(_dep_graph)
        _log_dependency_state(_dep_graph, _sched)
        _structured = build_structured_tool_results(context=merged_context, raw_results=task_results)

        hitl_question = (
            final_dependency_result.hitl_question
            or _DEFAULT_HITL_QUESTION
        )
        updates["planner_status"] = PlannerStatus.MISSING_REQUIRED_INFO.value
        updates["planner_task_results"] = task_results
        updates["planner_structured_results"] = _structured.model_dump()
        updates["planner_dependency_graph"] = _dep_graph.model_dump()
        updates["planner_scheduler_result"] = _sched.model_dump()
        updates["awaiting_user_clarification"] = True
        updates["pending_trip_context"] = merged_context.model_dump()
        updates["pending_missing_fields"] = [
            item.field_name for item in final_dependency_result.missing_requirements
        ]
        updates["pending_hitl_question"] = hitl_question
        updates["pending_planner_task_results"] = task_results
        updates["messages"] = [AIMessage(content=hitl_question)]

        logger.info("Master planner stopped for HITL: %s", hitl_question)
        return updates

    # Happy path: web fallback → cost calc → single dependency graph computation.
    task_results = await fill_missing_with_web(merged_context, task_results)

    cost_result = await calculate_cost_if_possible(
        context=merged_context,
        task_results=task_results,
    )
    if cost_result:
        task_results[PlannerTaskType.CALCULATE_TRIP_COST.value] = cost_result

    # P1-4.4: Single authoritative computation — runs once, after all data is final.
    completed_tasks = completed_tasks_from(task_results)
    dependency_graph = build_planner_dependency_graph(context=merged_context, completed_tasks=completed_tasks)
    scheduler_result = build_scheduler_result(dependency_graph)
    _log_dependency_state(dependency_graph, scheduler_result)
    structured_results = build_structured_tool_results(context=merged_context, raw_results=task_results)

    updates["planner_task_results"] = task_results
    updates["planner_structured_results"] = structured_results.model_dump()
    updates["planner_dependency_graph"] = dependency_graph.model_dump()
    updates["planner_scheduler_result"] = scheduler_result.model_dump()
    updates["planner_status"] = final_dependency_result.status.value

    # ── Budget gate: skip plan generation if cheapest options exceed budget ────
    _over_budget = _check_budget_exceeded(merged_context, cost_result)
    updates["over_budget"] = _over_budget
    if _over_budget:
        _budget_msg = _build_over_budget_message(merged_context, cost_result)
        logger.info(
            "Master planner: over-budget detected. budget=%.2f",
            merged_context.total_budget or 0,
        )
        updates["planner_status"] = PlannerStatus.READY.value
        updates["messages"] = [AIMessage(content=_budget_msg)]
        updates["used_web_source"] = False
        updates["final_plan"] = {}
        updates["tool_call_count"] = state.get("tool_call_count", 0) + len(task_results)
        updates["awaiting_user_clarification"] = False
        updates["pending_trip_context"] = {}
        updates["pending_missing_fields"] = []
        updates["pending_hitl_question"] = ""
        updates["pending_planner_task_results"] = {}
        updates["force_replan"] = False
        updates["hitl_feedback"] = ""
        updates["hitl_decision"] = ""
        return updates

    final_answer, final_plan = await generate_final_plan(
        context=merged_context,
        dependency_result=final_dependency_result,
        task_results=task_results,
        planning_mode=planning_mode,
        hitl_feedback=hitl_feedback,
        critic_issues=critic_issues,
        critic_suggestions=critic_suggestions,
    )

    used_web_source = final_plan.used_web_source or any(
        isinstance(v, str) and v.startswith("[Web source]")
        for v in task_results.values()
    )
    updates["used_web_source"] = used_web_source
    updates["final_plan"] = final_plan.model_dump()

    updates["planner_status"] = PlannerStatus.READY.value
    updates["messages"] = [AIMessage(content=final_answer)]
    updates["tool_call_count"] = state.get("tool_call_count", 0) + len(task_results)

    updates["awaiting_user_clarification"] = False
    updates["pending_trip_context"] = {}
    updates["pending_missing_fields"] = []
    updates["pending_hitl_question"] = ""
    updates["pending_planner_task_results"] = {}
    updates["force_replan"] = False
    updates["hitl_feedback"] = ""
    updates["hitl_decision"] = ""
    # Do NOT reset critic_attempts here — the critic node increments it and the
    # router uses it to cap the replan loop. Resetting here would cause an
    # infinite loop (planner always resets to 0, critic always sees attempt 1).

    logger.info("Master planner completed final plan. planning_mode=%s", planning_mode)
    return updates


_db_supervisor = DBSupervisor()
_web_supervisor = WebSupervisor(db_supervisor=_db_supervisor)


async def run_sub_agents_async(
    context: TripContext,
    existing_results: Optional[Dict[str, str]] = None,
    allowed_tasks: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """
    Delegates to WebSupervisor, which coordinates both tiers:
      - DBSupervisor  → Tier 1 DB agents (TransportAgent, StayAgent, ExperienceAgent)
      - WebSupervisor → Tier 2 web agents (TransportWebAgent, StayWebAgent,
                        ExperienceWebAgent, ManagerWebAgent)
    CyberAgent Zero-Trust steps 1-2 (outbound) and 4-6 (inbound) wrap both tiers.
    """
    return await _web_supervisor.dispatch(
        context=context,
        existing_results=existing_results,
        allowed_tasks=allowed_tasks,
    )


_MAX_TRAVEL_PREF_ENTRIES = 10


def _build_preference_state_updates(
    state: AgentState,
    preference_updates: List[PreferenceUpdate],
) -> dict:
    updates: dict = {}

    for update in preference_updates:
        if update.field_name == "travel_preferences":
            existing = state.get("travel_preferences", "") or ""
            pending = updates.get("travel_preferences", existing) or ""

            raw = (
                f"{pending}\n- {update.value}".strip()
                if pending
                else f"- {update.value}"
            )

            # Deduplicate and cap at _MAX_TRAVEL_PREF_ENTRIES bullet points.
            lines = [l for l in raw.splitlines() if l.strip()]
            seen: set = set()
            deduped: List[str] = []
            for line in lines:
                key = line.strip().lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(line)
            if len(deduped) > _MAX_TRAVEL_PREF_ENTRIES:
                deduped = deduped[-_MAX_TRAVEL_PREF_ENTRIES:]  # keep most recent
                logger.info(
                    "travel_preferences. trimmed_to=%d entries", _MAX_TRAVEL_PREF_ENTRIES
                )

            updates["travel_preferences"] = "\n".join(deduped)
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


_DEFAULT_HITL_QUESTION = (
    "I can plan this trip, but I need the origin airport, origin country, "
    "destination city, trip duration, and total budget first."
)


def _check_budget_exceeded(context: TripContext, cost_result: Optional[str]) -> bool:
    """Returns True when the cheapest estimated trip cost exceeds the user's budget."""
    if context.total_budget is None or not cost_result:
        return False
    try:
        import json as _j
        cost = _j.loads(cost_result)
        total_str = (
            cost.get("total_estimate", "")
            or cost.get("total_estimated", "")
            or cost.get("total", "")
        )
        if not total_str:
            return False
        total_val = float(str(total_str).lstrip("$").replace(",", ""))
        return total_val > context.total_budget
    except (ValueError, TypeError, KeyError):
        return False


def _build_over_budget_message(context: TripContext, cost_result: Optional[str]) -> str:
    """Builds a clear over-budget message shown instead of the full plan."""
    budget_str = (
        f"${context.total_budget:,.0f} {context.currency or 'USD'}"
        if context.total_budget else "your stated budget"
    )
    total_str = "—"
    flight_str = ""
    hotel_str = ""
    if cost_result:
        try:
            import json as _j
            cost = _j.loads(cost_result)
            total_str = (
                cost.get("total_estimate", "")
                or cost.get("total_estimated", "")
                or cost.get("total", "")
                or "—"
            )
            flight_str = cost.get("flight", "")
            hotel_str = cost.get("hotel", "")
        except (ValueError, TypeError):
            pass

    breakdown_lines = ""
    if flight_str:
        breakdown_lines += f"\n- Flight: {flight_str}"
    if hotel_str:
        breakdown_lines += f"\n- Hotel: {hotel_str}"
    if breakdown_lines:
        breakdown_lines = f"\n\n**Cost breakdown:**{breakdown_lines}"

    return (
        f"## Budget Alert — Cheapest Options Exceed Your Budget\n\n"
        f"The estimated minimum cost for this trip is **{total_str}**, "
        f"which exceeds your budget of **{budget_str}**.{breakdown_lines}\n\n"
        f"To continue, please:\n"
        f"- **[E] Edit** — increase your budget, shorten the trip, or choose a closer destination\n"
        f"- **[C] Cancel** — discard and start a new search\n\n"
        f"_Note: [A] Approve is not available for over-budget trips._"
    )