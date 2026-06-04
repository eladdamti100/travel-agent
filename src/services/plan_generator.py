"""
Plan generator — LLM-powered Section 3 and final plan assembly.

Extracted from planner.py. The only module in the services layer that calls an LLM.
Sections 1 and 2 are built deterministically by plan_formatter; this module
adds the Notes section and assembles the three-section final answer.
"""

from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_model
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
) -> str:
    """
    Builds the final plan in three guaranteed sections.

    Sections 1 and 2 are assembled deterministically from parsed tool data.
    Section 3 (Notes) uses a focused LLM call for brief reasoning text.
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
    return f"{section1}{sep}{section2}{sep}{section3}"


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
    return f"# Section 3 — Notes and Assumptions\n\n{content}"
