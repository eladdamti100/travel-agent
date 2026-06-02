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

    else:
        deterministic_context = extract_trip_context_deterministic(state)
        logger.info("Master planner created fresh deterministic context.")

    enrichment_task = asyncio.create_task(
        enrich_trip_context_async(state, deterministic_context)
    )

    allowed_tasks: Optional[Set[str]] = None

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
    Builds the final plan in three guaranteed sections.

    Sections 1 and 2 are assembled deterministically from parsed tool data so
    they always appear in full regardless of LLM output length or model size.
    Section 3 (Notes) uses a focused LLM call for the brief reasoning text.
    """
    db_results = {k: v for k, v in task_results.items() if k in _DB_TASK_KEYS}
    web_results = {k: v for k, v in task_results.items() if k in _WEB_TASK_KEYS}

    section1 = _build_db_section(context, db_results)
    section2 = _build_web_section(web_results)
    section3 = await _generate_notes_section(
        context=context,
        dependency_result=dependency_result,
        db_results=db_results,
        web_results=web_results,
        planning_mode=planning_mode,
    )

    _sep = "\n\n---\n\n"
    return f"{section1}{_sep}{section2}{_sep}{section3}"


def _build_db_section(context: TripContext, db_results: Dict[str, str]) -> str:
    """Builds Section 1 (Database Data) deterministically from SQLite tool results."""
    parts: List[str] = ["# Section 1 — Database Data\n"]

    # Trip Summary
    budget_str = (
        f"${context.total_budget:,.2f} {context.currency or 'USD'}"
        if context.total_budget else "—"
    )
    parts.append("**Trip Summary**")
    parts.append(f"- Origin: {context.origin_airport or '—'} ({context.origin_country or '—'})")
    parts.append(f"- Destination: {context.destination_city or '—'}, {context.destination_country or '—'}")
    parts.append(f"- Duration: {context.duration_days or '—'} days")
    if context.travel_month:
        parts.append(f"- Travel month: {context.travel_month.capitalize()}")
    parts.append(f"- Total budget: {budget_str}")
    if context.travel_style:
        parts.append(f"- Travel style: {context.travel_style.capitalize()}")
    parts.append("")

    # Flights
    parts.append("**Flights**")
    flights_raw = db_results.get("fetch_flights", "")
    if flights_raw:
        try:
            flights = json.loads(flights_raw)
            if isinstance(flights, list) and flights:
                for f in flights[:5]:
                    parts.append(
                        f"- {f.get('airline', '—')}: {f.get('flight_number', '—')}, "
                        f"${f.get('price', '—')}"
                    )
            else:
                parts.append(f"- {flights_raw}")
        except (json.JSONDecodeError, TypeError):
            parts.append(f"- {flights_raw}")
    else:
        parts.append("- Not available")
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
    has_experience = False

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
            parts.extend(lines)
            if lines:
                has_experience = True

    if not has_experience:
        parts.append("- Not available")
    parts.append("")

    # Visa
    parts.append("**Visa Information**")
    visa_raw = db_results.get("check_visa", "")
    parts.append(f"- {visa_raw}" if visa_raw else "- Not available — check the official embassy website.")
    parts.append("")

    # Cost Summary
    parts.append("**Cost Summary**")
    cost_raw = db_results.get("calculate_trip_cost", "")
    if cost_raw:
        try:
            cost = json.loads(cost_raw)
            for k, v in cost.items():
                if k != "currency":
                    parts.append(f"- {k.replace('_', ' ').title()}: {v}")
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


def _build_web_section(web_results: Dict[str, str]) -> str:
    """Builds Section 2 (Live Web Data) deterministically from WebAgent results."""
    parts: List[str] = ["# Section 2 — Live Web Data\n"]
    has_any = False

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

    # Live events — skip lines with unknown price
    events_raw = web_results.get("fetch_live_events", "")
    if events_raw:
        parts.append("**Live Events**")
        for line in events_raw.strip().split("\n")[:5]:
            stripped = line.strip().lstrip("- ")
            if stripped:
                parts.append(f"- {stripped}")
        parts.append("")
        has_any = True

    # Local breweries — deduplicate by name
    brew_raw = web_results.get("fetch_breweries", "")
    if brew_raw:
        parts.append("**Local Breweries & Pubs**")
        seen_names: set = set()
        count = 0
        for line in brew_raw.strip().split("\n"):
            stripped = line.strip().lstrip("- ")
            if not stripped:
                continue
            name_key = stripped.split("|")[0].strip().lower()
            if name_key in seen_names:
                continue
            seen_names.add(name_key)
            parts.append(f"- {stripped}")
            count += 1
            if count >= 3:
                break
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


def _parse_tavily_bullets(raw: str, max_bullets: int = 4) -> List[str]:
    """
    Strips 'Source: URL' lines from a Tavily result and returns
    the content as a list of clean bullet-point strings.
    """
    import re as _re

    sentences: List[str] = []
    for block in raw.split("\n\n"):
        for line in block.split("\n"):
            line = line.strip()
            if not line or line.lower().startswith("source:"):
                continue
            if line.lower().startswith("content:"):
                line = line[len("content:"):].strip()
            if line:
                # Split on sentence boundaries
                for s in _re.split(r"(?<=[.!?])\s+", line):
                    s = s.strip()
                    if len(s) > 30:
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
) -> str:
    """Asks the LLM for Section 3 (Notes and Assumptions) only — a focused, short call."""
    model = get_model(temperature=0)

    cost_raw = db_results.get("calculate_trip_cost", "")
    missing_fields = [r.field_name for r in dependency_result.missing_requirements]

    response = await model.ainvoke([
        SystemMessage(content=get_prompt("final_answer_prompt")),
        HumanMessage(
            content=(
                f"Planning mode: {planning_mode}\n"
                f"Traveler budget: ${context.total_budget} {context.currency or 'USD'}\n"
                f"Cost result: {cost_raw or 'not calculated'}\n"
                f"Missing required fields: {missing_fields or 'none'}\n"
                f"DB data collected: {sorted(db_results.keys())}\n"
                f"Web data collected: {sorted(web_results.keys())}"
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