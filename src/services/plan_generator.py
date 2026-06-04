"""
Plan generator — LLM-powered Section 3 and final plan assembly.

Extracted from planner.py. The only module in the services layer that calls an LLM.
Sections 1 and 2 are built deterministically by plan_formatter; this module
adds the Notes section and assembles the three-section final answer.

Returns both a markdown string (for the terminal UI) and a FinalPlan object
(for programmatic consumers such as the FastAPI layer).
"""

import json
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_model
from src.config.city_registry import CITY_BY_AIRPORT as _AIRPORT_CITY
from src.models.final_plan import (
    ActivityItem,
    CostSummary,
    FinalPlan,
    FlightOption,
    HotelOption,
    TripSummary,
    WebEnrichment,
)
from src.models.planner import DependencyCheckResult, PlannerTaskType
from src.models.trip_context import TripContext
from src.prompts.loader import get_prompt
from src.services.plan_formatter import build_db_section, build_web_section
from src.utils.logger import get_logger
from src.utils.token_tracker import log_token_usage

logger = get_logger("plan_generator")

# Task key sets — mirrored from planner.py for section splitting.
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

_WEB_TASK_KEYS: frozenset = frozenset({
    PlannerTaskType.GEOCODE_LOCATION.value,
    PlannerTaskType.FETCH_LIVE_EVENTS.value,
    PlannerTaskType.LIVE_CURRENCY_CONVERSION.value,
    PlannerTaskType.FETCH_BREWERIES.value,
    PlannerTaskType.FETCH_COUNTRY_METADATA.value,
    PlannerTaskType.WEB_RESEARCH_TAVILY.value,
})


async def generate_final_plan(
    context: TripContext,
    dependency_result: DependencyCheckResult,
    task_results: Dict[str, str],
    planning_mode: str = "full_planning",
    hitl_feedback: str = "",
    critic_issues: Optional[List[str]] = None,
    critic_suggestions: Optional[List[str]] = None,
) -> Tuple[str, FinalPlan]:
    """
    Builds the final plan in three guaranteed sections.

    Sections 1 and 2 are assembled deterministically from parsed tool data.
    Section 3 (Notes) uses a focused LLM call for brief reasoning text.

    Returns (markdown_string, FinalPlan) so callers can choose their rendering.
    """
    db_results = {k: v for k, v in task_results.items() if k in _DB_TASK_KEYS}
    web_results = {k: v for k, v in task_results.items() if k in _WEB_TASK_KEYS}

    # Promote web-fallback DB results into Section 2 display.
    for key in (PlannerTaskType.FETCH_FLIGHTS.value, PlannerTaskType.CHECK_VISA.value):
        val = db_results.get(key, "")
        if val and val.startswith("[Web source]"):
            web_results = {**web_results, key: val}

    section1 = build_db_section(context, db_results)
    section2 = build_web_section(web_results)
    section3 = await generate_notes_section(
        context=context,
        dependency_result=dependency_result,
        db_results=db_results,
        web_results=web_results,
        planning_mode=planning_mode,
        hitl_feedback=hitl_feedback,
        critic_issues=critic_issues or [],
        critic_suggestions=critic_suggestions or [],
    )

    sep = "\n\n---\n\n"
    markdown = f"{section1}{sep}{section2}{sep}{section3}"

    structured = _build_final_plan(
        context=context,
        db_results=db_results,
        web_results=web_results,
        notes=section3,
        planning_mode=planning_mode,
        markdown=markdown,
        used_web=bool(web_results),
    )

    return markdown, structured


async def generate_notes_section(
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
    """Asks the LLM for Section 3 (Notes and Assumptions) — a focused, short call."""
    model = get_model(temperature=0)

    cost_raw = db_results.get("calculate_trip_cost", "")
    missing_fields = [r.field_name for r in dependency_result.missing_requirements]

    feedback_line = (
        f"User requested changes: {hitl_feedback}\n"
        if hitl_feedback
        else ""
    )

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
    log_token_usage(response, call_site="plan_generator.notes_section")

    content = response.content if isinstance(response.content, str) else str(response.content)

    # Guard: empty or suspiciously short responses indicate a model failure.
    # Retry once with temperature=0.3 before falling back to a static message.
    _MIN_NOTES_CHARS = 50
    _MAX_NOTES_CHARS = 3000
    if len(content.strip()) < _MIN_NOTES_CHARS:
        logger.warning(
            "plan_generator. notes_section_too_short=%d retrying", len(content.strip())
        )
        try:
            retry_model = get_model(temperature=0.3)
            retry_resp = await retry_model.ainvoke([
                SystemMessage(content=get_prompt("final_answer_prompt")),
                HumanMessage(content=(
                    f"Planning mode: {planning_mode}\n"
                    f"Traveler budget: ${context.total_budget} {context.currency or 'USD'}\n"
                    f"DB tool results (key: status): {db_status}\n"
                    f"Web tool results (key: status): {web_status}"
                )),
            ])
            retry_content = retry_resp.content if isinstance(retry_resp.content, str) else str(retry_resp.content)
            if len(retry_content.strip()) >= _MIN_NOTES_CHARS:
                content = retry_content
                log_token_usage(retry_resp, call_site="plan_generator.notes_section_retry")
        except Exception as exc:
            logger.error("plan_generator. notes_section_retry_failed=%s", exc)

    # Truncate runaway responses to prevent the final answer from being too long.
    if len(content) > _MAX_NOTES_CHARS:
        content = content[:_MAX_NOTES_CHARS] + "\n\n*(response truncated)*"

    if len(content.strip()) < _MIN_NOTES_CHARS:
        content = "Notes could not be generated. Please review Sections 1 and 2 for your trip details."

    return f"# Section 3 — Notes and Assumptions\n\n{content}"


# ── FinalPlan builder ─────────────────────────────────────────────────────────

def _build_final_plan(
    context: TripContext,
    db_results: Dict[str, str],
    web_results: Dict[str, str],
    notes: str,
    planning_mode: str,
    markdown: str,
    used_web: bool,
) -> FinalPlan:
    """Assembles a FinalPlan from parsed tool results. Never raises."""
    try:
        airport = context.origin_airport or ""
        trip_summary = TripSummary(
            origin_airport=airport,
            origin_city=_AIRPORT_CITY.get(airport.upper(), context.origin_country),
            destination_city=context.destination_city,
            destination_country=context.destination_country,
            duration_days=context.duration_days,
            travel_month=context.travel_month,
            total_budget=context.total_budget,
            currency=context.currency or "USD",
            travel_style=context.travel_style,
        )

        flights: List[FlightOption] = []
        raw_flights = db_results.get(PlannerTaskType.FETCH_FLIGHTS.value, "")
        if raw_flights and not raw_flights.startswith("[Web source]"):
            try:
                for f in json.loads(raw_flights)[:5]:
                    flights.append(FlightOption(
                        airline=f.get("airline", ""),
                        flight_number=f.get("flight_number", ""),
                        price=f.get("price"),
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        hotels: List[HotelOption] = []
        raw_hotels = db_results.get(PlannerTaskType.FETCH_HOTELS.value, "")
        if raw_hotels:
            try:
                for h in json.loads(raw_hotels)[:5]:
                    hotels.append(HotelOption(
                        name=h.get("name", ""),
                        price_per_night=h.get("price_per_night"),
                        stars=h.get("stars"),
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        activities: List[ActivityItem] = []
        raw_acts = db_results.get(PlannerTaskType.FETCH_ACTIVITIES.value, "")
        if raw_acts:
            try:
                for a in json.loads(raw_acts)[:5]:
                    activities.append(ActivityItem(
                        name=a.get("name", ""),
                        category=a.get("category", ""),
                        price=a.get("price"),
                    ))
            except (json.JSONDecodeError, TypeError):
                pass

        # Cost summary
        cost_summary = CostSummary(currency=context.currency or "USD")
        raw_cost = db_results.get(PlannerTaskType.CALCULATE_TRIP_COST.value, "")
        if raw_cost:
            try:
                c = json.loads(raw_cost)
                cost_summary.flight_cost = c.get("flight_cost")
                cost_summary.hotel_total = c.get("hotel_total")
                cost_summary.estimated_total = c.get("total_cost") or c.get("total")
                if cost_summary.estimated_total and context.total_budget:
                    cost_summary.within_budget = cost_summary.estimated_total <= context.total_budget
            except (json.JSONDecodeError, TypeError):
                pass

        # Web enrichment
        web_enrichment = WebEnrichment()
        raw_geo = web_results.get(PlannerTaskType.GEOCODE_LOCATION.value, "")
        if raw_geo:
            try:
                geo = json.loads(raw_geo)
                web_enrichment.coordinates = {"lat": geo.get("lat", 0), "lng": geo.get("lng", 0)}
            except (json.JSONDecodeError, TypeError):
                pass

        raw_curr = web_results.get(PlannerTaskType.LIVE_CURRENCY_CONVERSION.value, "")
        if raw_curr:
            try:
                curr = json.loads(raw_curr)
                web_enrichment.currency_rate = (
                    f"{curr.get('original', '')} = {curr.get('converted', '')}"
                )
            except (json.JSONDecodeError, TypeError):
                pass

        raw_events = web_results.get(PlannerTaskType.FETCH_LIVE_EVENTS.value, "")
        if raw_events:
            web_enrichment.live_events = [
                l.strip().lstrip("- ") for l in raw_events.splitlines()
                if l.strip()
            ][:5]

        raw_tavily = web_results.get(PlannerTaskType.WEB_RESEARCH_TAVILY.value, "")
        if raw_tavily and not raw_tavily.startswith("Search Engine"):
            web_enrichment.web_highlights = [
                l.strip().lstrip("- ") for l in raw_tavily.splitlines()
                if l.strip() and not l.strip().lower().startswith("source:")
            ][:4]

        visa_info = db_results.get(PlannerTaskType.CHECK_VISA.value, "")
        if visa_info and visa_info.startswith("[Web source]"):
            visa_info = visa_info.removeprefix("[Web source]").strip()

        return FinalPlan(
            trip_summary=trip_summary,
            flights=flights,
            hotels=hotels,
            activities=activities,
            visa_info=visa_info,
            cost_summary=cost_summary,
            web_enrichment=web_enrichment,
            notes=notes,
            planning_mode=planning_mode,
            used_web_source=used_web,
            raw_markdown=markdown,
        )

    except Exception as exc:
        logger.error("plan_generator. build_final_plan_failed=%s", exc)
        return FinalPlan(raw_markdown=markdown, planning_mode=planning_mode)
