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
from src.agents.sub_agents.web_agent import WebAgent
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

# Keys produced by SQLite-backed sub-agents (Section 1 in the final plan)
_DB_TASK_KEYS: frozenset = frozenset({
    PlannerTaskType.FETCH_FLIGHTS.value,
    PlannerTaskType.FETCH_HOTELS.value,
    PlannerTaskType.FETCH_ACTIVITIES.value,
    PlannerTaskType.CHECK_VISA.value,
    PlannerTaskType.CALCULATE_TRIP_COST.value,
    PlannerTaskType.FETCH_RESTAURANTS.value,
    PlannerTaskType.LOCAL_TRANSPORT_GUIDE.value,
    PlannerTaskType.FETCH_WEATHER.value,
    PlannerTaskType.EVENTS_FINDER.value,
    PlannerTaskType.AIRPORT_TRANSFER_INFO.value,
})

# Keys produced by WebAgent / live APIs (Section 2 in the final plan)
_WEB_TASK_KEYS: frozenset = frozenset({
    PlannerTaskType.GEOCODE_LOCATION.value,
    PlannerTaskType.FETCH_LIVE_EVENTS.value,
    PlannerTaskType.LIVE_CURRENCY_CONVERSION.value,
    PlannerTaskType.FETCH_BREWERIES.value,
    PlannerTaskType.FETCH_COUNTRY_METADATA.value,
    PlannerTaskType.WEB_RESEARCH_TAVILY.value,
})


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

    # When replanning from HITL edit feedback, parse the feedback string for
    # updated trip parameters (duration, budget, origin, destination) and apply
    # them directly so the new plan reflects what the user asked to change.
    if state.get("force_replan") and hitl_feedback:
        import re as _re
        fb_lower = hitl_feedback.lower()

        _CITY_TO_AIRPORT_MAP = {
            "paris": "CDG", "london": "LHR", "new york": "JFK",
            "tokyo": "NRT", "berlin": "BER", "tel aviv": "TLV",
        }

        # Origin airport/city override — patterns like:
        #   "from paris", "flight from paris", "origin is paris", "flight is from paris"
        _origin_triggers = (
            r"(?:flight(?:\s+is)?|fly(?:ing)?|depart(?:ing)?|origin)\s+(?:is\s+)?from\s+",
            r"\bfrom\s+",
            r"\borigin\s+(?:is\s+)?(?:city\s+)?(?:is\s+)?",
            r"\bchange\s+(?:the\s+)?(?:flight\s+)?(?:origin|departure)\s+to\s+",
        )
        fb_origin_airport = None
        fb_origin_city = None

        # First try: bare IATA code in the feedback (e.g. "change to CDG")
        _iata_match = _re.search(r"\b([A-Z]{3})\b", hitl_feedback.upper())
        if _iata_match:
            _iata = _iata_match.group(1)
            _iata_to_city = {v: k.title() for k, v in _CITY_TO_AIRPORT_MAP.items()}
            if _iata in _iata_to_city:
                fb_origin_airport = _iata
                fb_origin_city = _iata_to_city[_iata]

        # Second try: city name after an origin-intent phrase
        if not fb_origin_airport:
            for _trigger in _origin_triggers:
                for _city, _code in _CITY_TO_AIRPORT_MAP.items():
                    if _re.search(_trigger + _re.escape(_city), fb_lower):
                        fb_origin_airport = _code
                        fb_origin_city = _city.title()
                        break
                if fb_origin_airport:
                    break

        if fb_origin_airport:
            old_airport = deterministic_context.origin_airport or "?"
            old_city = _AIRPORT_CITY.get(old_airport, old_airport)
            print(
                f"[HITL edit] origin change detected: {old_airport} ({old_city})"
                f" → {fb_origin_airport} ({fb_origin_city})"
            )
            logger.info(
                "HITL edit: origin override %s→%s", old_airport, fb_origin_airport
            )
            deterministic_context = deterministic_context.model_copy(
                update={"origin_airport": fb_origin_airport}
            )

        # Destination city override — "change destination to Berlin", "fly to Berlin"
        _dest_triggers = (
            r"(?:change\s+)?(?:the\s+)?destination\s+to\s+",
            r"\bfly\s+to\s+",
            r"\btravel\s+to\s+",
        )
        for _trigger in _dest_triggers:
            for _city, _code in _CITY_TO_AIRPORT_MAP.items():
                if _re.search(_trigger + _re.escape(_city), fb_lower):
                    from src.models.trip_context import DESTINATION_COUNTRY_BY_CITY
                    _dest_country = DESTINATION_COUNTRY_BY_CITY.get(_city.title())
                    deterministic_context = deterministic_context.model_copy(
                        update={
                            "destination_city": _city.title(),
                            "destination_country": _dest_country,
                        }
                    )
                    logger.info("HITL edit: destination override → %s", _city.title())
                    break

        # Duration override from feedback (e.g. "change to 14 days", "14-day trip")
        _dur_patterns = [
            r"\b(\d+)\s*-\s*day\b",
            r"\b(\d+)\s+days?\b",
            r"\b(\d+)\s+nights?\b",
            r"\bfor\s+(\d+)\s+days?\b",
            r"\bfor\s+(\d+)\s+nights?\b",
        ]
        fb_duration = None
        for _pat in _dur_patterns:
            _m = _re.search(_pat, fb_lower)
            if _m:
                fb_duration = int(_m.group(1))
                break

        if fb_duration:
            deterministic_context = deterministic_context.model_copy(
                update={"duration_days": fb_duration}
            )
            logger.info("Planner applied HITL feedback duration override: %d days", fb_duration)

        # Budget override from feedback (e.g. "change budget to $2000")
        budget_match = _re.search(r"\$(\d[\d,]*(?:\.\d+)?)", fb_lower)
        if budget_match:
            fb_budget = float(budget_match.group(1).replace(",", ""))
            deterministic_context = deterministic_context.model_copy(
                update={"total_budget": fb_budget}
            )
            logger.info("Planner applied HITL feedback budget override: $%.2f", fb_budget)

    enrichment_task = asyncio.create_task(
        enrich_trip_context_async(state, deterministic_context)
    )

    allowed_tasks: Optional[Set[str]] = None

    is_critic_replan = (state.get("critic_attempts") or 0) > 0 and not is_hitl_resume

    # On automatic critic replan, surface the critic's issues and suggestions so
    # the LLM can address them in the new plan.  These are separate from the
    # human-provided hitl_feedback (which is only set on the Edit path).
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

    # Early-exit: destination not in supported list — no DB data exists for it.
    _SUPPORTED_DESTINATIONS = set(
        __import__("src.models.trip_context", fromlist=["DESTINATION_COUNTRY_BY_CITY"])
        .DESTINATION_COUNTRY_BY_CITY.keys()
    )
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

    # Web fallback: fill missing DB sections via Tavily search.
    task_results = await _fill_missing_with_web(merged_context, task_results)
    updates["planner_task_results"] = task_results

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
        hitl_feedback=hitl_feedback,
        critic_issues=critic_issues,
        critic_suggestions=critic_suggestions,
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
    updates["hitl_feedback"] = ""
    updates["hitl_decision"] = ""
    # Do NOT reset critic_attempts here — the critic node increments it and the
    # router uses it to cap the replan loop. Resetting here would cause an
    # infinite loop (planner always resets to 0, critic always sees attempt 1).

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
        for agent in [TransportAgent(), StayAgent(), ExperienceAgent(), WebAgent()]
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


async def _fill_missing_with_web(
    context: TripContext,
    task_results: Dict[str, str],
) -> Dict[str, str]:
    """
    For each DB section that returned no data, fires a targeted Tavily web search
    and stores the result under the same task key so the final plan can surface it.

    Runs all fallback queries concurrently. Failures are logged and silently skipped.
    """
    from src.tools.web_api_tools import web_research_tavily

    city = context.destination_city or ""
    origin = context.origin_airport or ""

    def _is_empty(key: str) -> bool:
        val = task_results.get(key, "")
        if not val:
            return True
        # Tool returns "No X found …" style strings for empty results.
        return val.strip().lower().startswith("no ")

    queries: Dict[str, str] = {}

    if _is_empty(PlannerTaskType.FETCH_FLIGHTS.value) and origin and city:
        queries[PlannerTaskType.FETCH_FLIGHTS.value] = (
            f"best flights from {origin} to {city} airlines prices schedule"
        )

    if _is_empty(PlannerTaskType.FETCH_HOTELS.value) and city:
        queries[PlannerTaskType.FETCH_HOTELS.value] = (
            f"best hotels to stay in {city} price per night budget"
        )

    if _is_empty(PlannerTaskType.FETCH_ACTIVITIES.value) and city:
        queries[PlannerTaskType.FETCH_ACTIVITIES.value] = (
            f"top tourist attractions and activities in {city} with prices"
        )

    if _is_empty(PlannerTaskType.CHECK_VISA.value) and context.origin_country and city:
        queries[PlannerTaskType.CHECK_VISA.value] = (
            f"visa requirements for {context.origin_country} passport to visit {city}"
        )

    if not queries:
        return task_results

    logger.info("Web fallback triggered for missing DB keys: %s", list(queries.keys()))

    async def _fetch(key: str, query: str) -> tuple:
        try:
            result = await web_research_tavily.ainvoke({"query": query})
            return key, str(result)
        except Exception as exc:
            logger.warning("Web fallback failed for key=%s: %s", key, exc)
            return key, ""

    pairs = await asyncio.gather(*[_fetch(k, q) for k, q in queries.items()])
    updated = dict(task_results)
    for key, result in pairs:
        if result and not result.startswith("Search Engine"):
            updated[key] = f"[Web source] {result}"
            logger.info("Web fallback stored result for key=%s (%d chars)", key, len(result))

    return updated


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

    if hotel_price is None:
        logger.info(
            "Cost calculation skipped: no hotel data. flight_price=%s duration_days=%s",
            flight_price,
            context.duration_days,
        )
        return None

    if flight_price is None:
        logger.info(
            "Cost calculation: no flight data — using 0 as flight cost placeholder."
        )
        flight_price = 0.0

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
    hitl_feedback: str = "",
    critic_issues: Optional[List[str]] = None,
    critic_suggestions: Optional[List[str]] = None,
) -> str:
    """
    Builds the final plan in three guaranteed sections.

    Sections 1 and 2 are assembled deterministically from parsed tool data so
    they always appear in full regardless of LLM output length or model size.
    Section 3 (Notes) uses a focused LLM call for the brief reasoning text.
    """
    db_results = {k: v for k, v in task_results.items() if k in _DB_TASK_KEYS}
    web_results = {k: v for k, v in task_results.items() if k in _WEB_TASK_KEYS}

    # Promote web-fallback DB results into web_results so Section 2 can display them
    for key in (PlannerTaskType.FETCH_FLIGHTS.value, PlannerTaskType.CHECK_VISA.value):
        val = db_results.get(key, "")
        if val and val.startswith("[Web source]"):
            web_results = {**web_results, key: val}

    section1 = _build_db_section(context, db_results)
    section2 = _build_web_section(web_results)
    section3 = await _generate_notes_section(
        context=context,
        dependency_result=dependency_result,
        db_results=db_results,
        web_results=web_results,
        planning_mode=planning_mode,
        hitl_feedback=hitl_feedback,
        critic_issues=critic_issues or [],
        critic_suggestions=critic_suggestions or [],
    )

    _sep = "\n\n---\n\n"
    return f"{section1}{_sep}{section2}{_sep}{section3}"


_AIRPORT_CITY: Dict[str, str] = {
    "JFK": "New York", "LGA": "New York", "EWR": "New York",
    "LAX": "Los Angeles", "ORD": "Chicago", "ATL": "Atlanta",
    "DFW": "Dallas", "SFO": "San Francisco", "MIA": "Miami",
    "LHR": "London", "LGW": "London", "CDG": "Paris", "ORY": "Paris",
    "TLV": "Tel Aviv", "NRT": "Tokyo", "HND": "Tokyo",
    "TXL": "Berlin", "BER": "Berlin",
}


def _build_db_section(context: TripContext, db_results: Dict[str, str]) -> str:
    """Builds Section 1 (Database Data) deterministically from SQLite tool results."""
    parts: List[str] = ["# Section 1 — Database Data\n"]

    # Trip Summary
    budget_str = (
        f"${context.total_budget:,.2f} {context.currency or 'USD'}"
        if context.total_budget else "—"
    )

    # Show airport code + origin city (not passport country)
    airport = context.origin_airport or "—"
    origin_city = _AIRPORT_CITY.get(airport.upper(), context.origin_country or "—")

    parts.append("**Trip Summary**")
    parts.append(f"- Origin: {airport} ({origin_city})")
    parts.append(f"- Destination: {context.destination_city or '—'}, {context.destination_country or '—'}")
    parts.append(f"- Duration: {context.duration_days or '—'} days")
    if context.travel_month:
        parts.append(f"- Travel month: {context.travel_month.capitalize()}")
    parts.append(f"- Total budget: {budget_str}")
    if context.travel_style:
        parts.append(f"- Travel style: {context.travel_style.capitalize()}")
    parts.append("")

    # Flights — only show confirmed DB records here; web fallback appears in Section 2
    parts.append("**Flights**")
    flights_raw = db_results.get("fetch_flights", "")
    flights_is_web = flights_raw.startswith("[Web source]") if flights_raw else False
    if flights_raw and not flights_is_web:
        try:
            flights = json.loads(flights_raw)
            if isinstance(flights, list) and flights:
                for f in flights[:5]:
                    price = f.get("price")
                    price_str = f"${price}" if price else "price unavailable"
                    parts.append(
                        f"- {f.get('airline', '—')}: {f.get('flight_number', '—')}, "
                        f"{price_str}"
                    )
            else:
                parts.append("- No confirmed flight records in database — see web data below.")
        except (json.JSONDecodeError, TypeError):
            parts.append("- No confirmed flight records in database — see web data below.")
    else:
        parts.append("- No confirmed flight records in database — see web data below.")
    parts.append("")

    # Hotels
    parts.append("**Hotels**")
    hotels_raw = db_results.get("fetch_hotels", "")
    if hotels_raw:
        try:
            hotels = json.loads(hotels_raw)
            if isinstance(hotels, list) and hotels:
                for h in hotels[:5]:
                    price = h.get("price_per_night", "—")
                    price_str = f"${price:.2f}" if isinstance(price, (int, float)) else f"${price}"
                    parts.append(
                        f"- {h.get('name', '—')}: {price_str}/night "
                        f"({h.get('stars', '—')} stars)"
                    )
            else:
                parts.append(f"- {hotels_raw}")
        except (json.JSONDecodeError, TypeError):
            parts.append(f"- {hotels_raw}")
    else:
        parts.append("- Not available")
    parts.append("")

    # Activities + restaurants + weather + events + local transport + airport transfer
    parts.append("**Activities & Experience**")
    parts.append("")
    has_experience = False

    _CATEGORY_LABELS = {
        "fetch_activities":       "Activities",
        "fetch_restaurants":      "Restaurants",
        "fetch_weather":          "Weather",
        "events_finder":          "Events",
        "local_transport_guide":  "Local Transport",
        "airport_transfer_info":  "Airport Transfers",
    }

    for raw_key, formatter in [
        ("fetch_activities",    _fmt_activities),
        ("fetch_restaurants",   _fmt_restaurants),
        ("fetch_weather",       _fmt_weather),
        ("events_finder",       _fmt_events),
        ("local_transport_guide", _fmt_transport),
        ("airport_transfer_info", _fmt_airport),
    ]:
        raw = db_results.get(raw_key, "")
        if raw:
            lines = formatter(raw)
            if lines:
                label = _CATEGORY_LABELS.get(raw_key, raw_key)
                parts.append(f"*{label}*")
                parts.extend(lines)
                parts.append("")
                has_experience = True

    if not has_experience:
        parts.append("- Not available")
        parts.append("")

    # Visa — only show confirmed DB records; web fallback appears in Section 2
    parts.append("**Visa Information**")
    visa_raw = db_results.get("check_visa", "")
    visa_is_web = visa_raw.startswith("[Web source]") if visa_raw else False
    if visa_raw and not visa_is_web:
        parts.append(f"- {visa_raw}")
    else:
        parts.append("- No visa data in database — see web research section below.")
    parts.append("")

    # Cost Summary
    parts.append("**Cost Summary**")
    cost_raw = db_results.get("calculate_trip_cost", "")
    flights_raw_for_cost = db_results.get("fetch_flights", "")
    flight_is_estimated = not flights_raw_for_cost or flights_raw_for_cost.startswith("[Web source]")
    if cost_raw:
        try:
            cost = json.loads(cost_raw)
            for k, v in cost.items():
                if k == "currency":
                    continue
                label = k.replace('_', ' ').title()
                if k == "flight" and flight_is_estimated:
                    parts.append(f"- {label}: {v} (flight price not confirmed — estimate only)")
                else:
                    parts.append(f"- {label}: {v}")
        except (json.JSONDecodeError, TypeError):
            parts.append(f"- {cost_raw}")
    else:
        parts.append("- Cost calculation not available")

    return "\n".join(parts)


def _fmt_activities(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        out = []
        for a in items[:5]:
            price = a.get("price")
            price_str = f"${price:.0f}" if isinstance(price, (int, float)) and price else "Free"
            out.append(f"- {a.get('name', '—')} ({a.get('category', '—')}): {price_str}")
        return out
    except (json.JSONDecodeError, TypeError):
        return [f"- {raw}"] if raw and not raw.startswith("No ") else []


def _fmt_restaurants(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        out = []
        for r in items[:3]:
            out.append(
                f"- Restaurant: {r.get('name', '—')} ({r.get('cuisine', '—')}), "
                f"{r.get('price_range', '—')}, ★{r.get('rating', '—')}"
            )
        return out
    except (json.JSONDecodeError, TypeError):
        return []


def _fmt_weather(raw: str) -> List[str]:
    try:
        w = json.loads(raw)
        month = (w.get("month") or "").capitalize()
        return [
            f"- Weather ({month}): {w.get('avg_temp_c', '—')}°C / "
            f"{w.get('avg_temp_f', '—')}°F — {w.get('description', '—')}"
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def _fmt_events(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        return [
            f"- Event: {e.get('name', '—')} ({e.get('category', '—')})"
            for e in items[:3]
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def _fmt_transport(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        return [
            f"- Local transport: {m.get('mode', '—')} — {m.get('price_range', '—')}"
            for m in items[:3]
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def _fmt_airport(raw: str) -> List[str]:
    try:
        items = json.loads(raw)
        if not isinstance(items, list):
            return []
        out = []
        for t in items[:2]:
            price = t.get("price_usd", "—")
            price_str = f"~${price:.0f}" if isinstance(price, (int, float)) else f"~${price}"
            out.append(
                f"- Airport transfer: {t.get('mode', '—')}, "
                f"{t.get('duration', '—')}, {price_str}"
            )
        return out
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_visa_summary(raw: str) -> str:
    """
    Extracts one clean visa status sentence from raw web research text.
    Looks for sentences containing 'visa-free', 'no visa', 'visa required', or 'ETIAS'.
    Skips markdown table rows (lines with 2+ pipe characters).
    Falls back to a generic message with the source attribution.
    """
    import re as _re

    _VISA_KEYWORDS = ("visa-free", "no visa", "visa required", "etias", "visa on arrival")
    candidates = []

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip source/content markers
        if line.lower().startswith("source:") or line.lower().startswith("content:"):
            continue
        # Skip markdown table rows — they contain country comparisons not relevant to traveler
        if line.count("|") >= 2:
            continue
        low = line.lower()
        if any(kw in low for kw in _VISA_KEYWORDS):
            for s in _re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                if (
                    len(s) > 20
                    and s.count("|") < 2
                    and any(kw in s.lower() for kw in _VISA_KEYWORDS)
                ):
                    candidates.append(s)
                    break
        if candidates:
            break

    if candidates:
        return candidates[0] + " (Source: web research)"
    return "Visa requirements could not be determined — check the official embassy website."


def _build_web_section(web_results: Dict[str, str]) -> str:
    """Builds Section 2 (Live Web Data) deterministically from WebAgent results."""
    import datetime as _dt
    _today = _dt.date.today()

    parts: List[str] = ["# Section 2 — Live Web Data\n"]
    has_any = False

    # Flights web fallback (promoted from DB section when no DB record found)
    flights_web = web_results.get("fetch_flights", "")
    if flights_web and flights_web.startswith("[Web source]"):
        parts.append("**Flights (Web Research)**")
        parts.append("- No direct flight records found in database for this route.")
        parts.append("- Check current prices and availability on Google Flights, Skyscanner, or Kayak.")
        parts.append("")
        has_any = True

    # Visa web fallback — extract one clean summary line from web research
    visa_web = web_results.get("check_visa", "")
    if visa_web and visa_web.startswith("[Web source]"):
        clean = visa_web.removeprefix("[Web source]").strip()
        visa_line = _extract_visa_summary(clean)
        parts.append("**Visa Information (Web Research)**")
        parts.append(f"- {visa_line}")
        parts.append("")
        has_any = True

    # Location coordinates
    geo_raw = web_results.get("geocode_location", "")
    if geo_raw:
        try:
            geo = json.loads(geo_raw)
            source = " (live)" if geo.get("status") == "verified_live" else " (fallback)"
            parts.append(f"**Location Coordinates**{source}")
            parts.append(f"- Lat: {geo.get('lat', '—')}, Lng: {geo.get('lng', '—')}\n")
            has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    # Country metadata
    country_raw = web_results.get("fetch_country_metadata", "")
    if country_raw:
        try:
            meta = json.loads(country_raw)
            parts.append("**Country Metadata**")
            parts.append(f"- Country: {meta.get('canonical_name', '—')}")
            parts.append(f"- Local currency: {meta.get('currency_code', '—')}")
            parts.append(f"- Region: {meta.get('region', '—')}\n")
            has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    # Currency conversion — budget + inverse rate + passport currency
    currency_raw = web_results.get("live_currency_conversion", "")
    if currency_raw:
        try:
            curr = json.loads(currency_raw)
            source = " (live)" if curr.get("status") == "live_synchronized" else " (estimate)"
            dest_cur  = curr.get("destination_currency", "")
            pass_cur  = curr.get("passport_currency", "")

            parts.append(f"**Currency Conversion**{source}")
            parts.append(f"- Budget: {curr.get('original', '—')} = {curr.get('converted', '—')}")

            inv = curr.get("inverse_rate")
            if inv and dest_cur:
                parts.append(f"- 1 {dest_cur} = {inv} USD")

            one_in_pass = curr.get("one_dest_in_passport")
            if one_in_pass and dest_cur and pass_cur and pass_cur != "USD":
                parts.append(f"- 1 {dest_cur} = {one_in_pass} ({pass_cur})")

            budget_in_pass = curr.get("budget_in_passport")
            if budget_in_pass and pass_cur and pass_cur != "USD":
                parts.append(f"- Full budget in {pass_cur}: {budget_in_pass}")

            parts.append("")
            has_any = True
        except (json.JSONDecodeError, TypeError):
            pass

    # Live events — skip noisy or empty lines
    events_raw = web_results.get("fetch_live_events", "")
    if events_raw:
        event_lines = []
        for line in events_raw.strip().split("\n"):
            stripped = line.strip().lstrip("- ")
            if stripped and not _is_noisy_line(stripped):
                event_lines.append(f"- {stripped}")
            if len(event_lines) >= 5:
                break
        if event_lines:
            parts.append("**Live Events**")
            parts.extend(event_lines)
            parts.append("")
            has_any = True

    # Local breweries — deduplicate by name, skip noise
    brew_raw = web_results.get("fetch_breweries", "")
    if brew_raw:
        seen_names: set = set()
        brew_lines = []
        for line in brew_raw.strip().split("\n"):
            stripped = line.strip().lstrip("- ")
            if not stripped or _is_noisy_line(stripped):
                continue
            name_key = stripped.split("|")[0].strip().lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            brew_lines.append(f"- {stripped}")
            if len(brew_lines) >= 3:
                break
        if brew_lines:
            parts.append("**Local Breweries & Pubs**")
            parts.extend(brew_lines)
            parts.append("")
            has_any = True

    # Tavily web research — strip URLs, format as bullet points
    tavily_raw = web_results.get("web_research_tavily", "")
    if tavily_raw and not tavily_raw.startswith("Search Engine"):
        bullets = _parse_tavily_bullets(tavily_raw)
        if bullets:
            parts.append("**Web Research Highlights**")
            parts.extend(bullets)
            parts.append("")
            has_any = True

    if not has_any:
        parts.append("- No live web data was available for this trip.")

    return "\n".join(parts)


_NOISE_PATTERNS = (
    # image placeholders
    "image ",
    # navigation / UI labels scraped from booking sites
    "select dates",
    "weekly",
    "monthly",
    "view all",
    "load more",
    "read more",
    "click here",
    "sign up",
    "subscribe",
    "advertisement",
    # broken table artefacts — lines that are mostly pipes
)


def _is_noisy_line(text: str) -> bool:
    """Returns True when a line looks like UI junk or a scraping artefact."""
    low = text.strip().lower()
    if not low:
        return True
    # Starts with a noise keyword
    if any(low.startswith(p) for p in _NOISE_PATTERNS):
        return True
    # Line is mostly pipe characters (broken markdown table)
    if low.count("|") >= 3:
        return True
    return False


def _parse_tavily_bullets(raw: str, max_bullets: int = 4) -> List[str]:
    """
    Strips 'Source: URL' lines, UI-noise, and clearly outdated date references
    from a Tavily result, then returns clean bullet-point strings.
    """
    import re as _re
    import datetime as _dt

    _today = _dt.date.today()

    # Regex to detect a past year range like "September 23, 2025, to January 11, 2026"
    # or standalone past years embedded in a sentence.
    _past_year_re = _re.compile(r"\b(20\d{2})\b")

    def _sentence_is_stale(s: str) -> bool:
        """Returns True when a sentence mentions only past years (before today)."""
        years = [int(y) for y in _past_year_re.findall(s)]
        if not years:
            return False
        # Allow if any mentioned year is current or future
        return all(y < _today.year for y in years)

    sentences: List[str] = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            line = line.strip()
            if not line or line.lower().startswith("source:"):
                continue
            if line.lower().startswith("content:"):
                line = line[len("content:"):].strip()
            if _is_noisy_line(line):
                continue
            for s in _re.split(r"(?<=[.!?])\s+", line):
                s = s.strip()
                if len(s) > 30 and not _is_noisy_line(s) and not _sentence_is_stale(s):
                    sentences.append(s)

    # Deduplicate while preserving order
    seen: set = set()
    bullets: List[str] = []
    for s in sentences:
        key = s[:60].lower()
        if key not in seen:
            seen.add(key)
            bullets.append(f"- {s}")
        if len(bullets) >= max_bullets:
            break

    return bullets


async def _generate_notes_section(
    *,
    context: TripContext,
    dependency_result: DependencyCheckResult,
    db_results: Dict[str, str],
    web_results: Dict[str, str],
    planning_mode: str,
    hitl_feedback: str = "",
    critic_issues: Optional[List[str]] = None,
    critic_suggestions: Optional[List[str]] = None,
) -> str:
    """Asks the LLM for Section 3 (Notes and Assumptions) only — a focused, short call."""
    model = get_model(temperature=0)

    cost_raw = db_results.get("calculate_trip_cost", "")
    missing_fields = [r.field_name for r in dependency_result.missing_requirements]

    feedback_line = (
        f"User requested changes: {hitl_feedback}\n"
        if hitl_feedback
        else ""
    )

    # On automatic critic replan, surface issues and suggestions so the LLM
    # knows what specifically went wrong and what it must fix.
    critic_context_lines: list = []
    if critic_issues:
        critic_context_lines.append(
            "CRITIC REJECTED THE PREVIOUS PLAN — you MUST address these issues:"
        )
        for issue in critic_issues:
            critic_context_lines.append(f"  - {issue}")
    if critic_suggestions:
        critic_context_lines.append("Required fixes (apply all of them):")
        for suggestion in critic_suggestions:
            critic_context_lines.append(f"  → {suggestion}")
    critic_context = "\n".join(critic_context_lines) + "\n" if critic_context_lines else ""

    def _result_status(raw: str) -> str:
        if not raw:
            return "not_collected"
        stripped = raw.strip().lower()
        if stripped.startswith("no ") or stripped.startswith("error"):
            return "not_found"
        if stripped.startswith("[web source]"):
            return "web_fallback"
        return "found"

    db_status = {k: _result_status(v) for k, v in db_results.items()}
    web_status = {k: _result_status(v) for k, v in web_results.items()}

    response = await model.ainvoke([
        SystemMessage(content=get_prompt("final_answer_prompt")),
        HumanMessage(
            content=(
                f"Planning mode: {planning_mode}\n"
                f"{critic_context}"
                f"{feedback_line}"
                f"Traveler budget: ${context.total_budget} {context.currency or 'USD'}\n"
                f"Cost result: {cost_raw or 'not calculated'}\n"
                f"Missing required fields: {missing_fields or 'none'}\n"
                f"DB tool results (key: status): {db_status}\n"
                f"Web tool results (key: status): {web_status}"
            )
        ),
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)
    return f"# Section 3 — Notes and Assumptions\n\n{content}"


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