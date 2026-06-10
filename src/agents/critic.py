"""
Critic agent — deterministic plan quality and budget gate.

Pure logic module — no LangGraph code, no graph imports.
Student 4 imports critique_plan() and wires it into the graph.

What it checks
──────────────
1. Budget gate        — estimated total cost vs. stated budget.
                        Fails hard if the plan costs more than the user can spend.
2. Completeness gate  — plan must contain flights, hotels, and activities.
                        Soft fail — issues are reported but don't block the plan.
3. Suggestions        — when checks fail, suggestions are concrete enough for
                        the planner to act on immediately:
                        "find a hotel under $90/night" not "reduce costs".

Input
──────
state — the full AgentState dict.  Reads:
  state["total_budget"]                — user's stated budget (float | None)
  state["trip_context"]                — dict with duration_days, num_travelers, etc.
  state["planner_structured_results"]  — dict matching PlannerToolResults shape
  state["planner_task_results"]        — raw string results (fallback)

Output
──────
CritiqueResult dataclass — fully structured, ready for UI rendering.

Plan → Critique → Replan loop
──────────────────────────────
When passed=False, the caller (Student 4) can append suggestions to the
planner's next prompt so the planner replans with the constraints in mind.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.critic import BudgetBreakdown, CritiqueResult
from src.utils.logger import get_logger

logger = get_logger("critic")

# Budget allocation targets (percentage of total budget per category).
# Used to generate specific per-category suggestions when the total is exceeded.
_FLIGHT_BUDGET_SHARE = 0.35    # flights should consume at most 35% of budget
_HOTEL_BUDGET_SHARE  = 0.45    # hotel stay should consume at most 45% of budget
_ACTIVITIES_BUDGET_SHARE = 0.15  # activities ~15%, leaving 5% buffer

# Minimum score penalty per unresolved issue.
_SCORE_PENALTY_PER_ISSUE = 2
# Extra penalty when the plan exceeds the budget.
_BUDGET_OVERAGE_PENALTY = 3


# ── Cost extraction helpers ────────────────────────────────────────────────────

def _cheapest(items: List[dict], price_key: str) -> Optional[float]:
    """Returns the lowest non-null price from a list of result dicts."""
    prices = [
        item[price_key]
        for item in items
        if isinstance(item.get(price_key), (int, float))
    ]
    return min(prices) if prices else None


def _extract_cost_from_raw_text(raw_text: str) -> Optional[float]:
    """
    Fallback: scan raw planner output for a total cost figure.

    Looks for patterns like "Total: $2,450" or "estimated cost: 1800 USD".
    Returns the first matched number, or None if nothing found.
    """
    patterns = [
        r"total\s*(?:trip\s*)?cost[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"estimated\s+total[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"estimated\s*(?:total\s*)?(?:cost|price)[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"grand\s*total[:\s]+\$?([\d,]+(?:\.\d+)?)",
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(?:total|overall|in total)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _build_budget_breakdown(
    structured: dict,
    budget: Optional[float],
    duration_days: Optional[int],
    num_travelers: int,
) -> BudgetBreakdown:
    """
    Extracts and combines cost components from planner_structured_results.

    Priority order for total_cost:
      1. structured["cost"]["total_cost"]  — most accurate, set by calculate_trip_cost
      2. flight + hotel + activities       — computed from individual results
      3. raw text scan                     — last resort fallback
    """
    flights    = structured.get("flights", [])
    hotels     = structured.get("hotels", [])
    activities = structured.get("activities", [])
    cost_obj   = structured.get("cost") or {}

    # Per-person flight price (cheapest option)
    flight_pp = (
        cost_obj.get("flight_price")
        or _cheapest(flights, "price")
    )
    flight_cost = flight_pp * num_travelers if flight_pp else None

    # Hotel cost: cheapest nightly rate × duration
    hotel_nightly = (
        cost_obj.get("hotel_price_per_night")
        or _cheapest(hotels, "price_per_night")
    )
    hotel_cost = (
        hotel_nightly * duration_days
        if hotel_nightly and duration_days
        else None
    )

    # Activities: sum cheapest 3 options (per-person × travelers)
    activity_prices = sorted(
        [a["price"] for a in activities if isinstance(a.get("price"), (int, float))],
    )[:3]
    activities_cost = sum(activity_prices) * num_travelers if activity_prices else None

    # Total cost — use structured value first, then compute, then raw fallback
    total = cost_obj.get("total_cost")
    if not total:
        parts = [c for c in [flight_cost, hotel_cost, activities_cost] if c is not None]
        total = sum(parts) if parts else None

    overage = (total - budget) if (total and budget) else None
    within  = (total <= budget) if (total and budget) else None

    return BudgetBreakdown(
        budget=budget,
        flight_cost=flight_cost,
        hotel_cost=hotel_cost,
        activities_cost=activities_cost,
        total_estimated=total,
        overage=overage,
        within_budget=within,
    )


# ── Suggestion generators ──────────────────────────────────────────────────────

def _budget_suggestions(
    bd: BudgetBreakdown,
    duration_days: Optional[int],
) -> List[str]:
    """
    Returns concrete, actionable suggestions when the plan exceeds the budget.

    Each suggestion names a specific number — "under $90/night" not "cheaper hotel".
    """
    if not bd.budget or not bd.total_estimated or bd.within_budget:
        return []

    suggestions: List[str] = []
    budget = bd.budget
    overage = bd.overage or 0

    # Hotel suggestion
    if bd.hotel_cost and duration_days:
        current_nightly = bd.hotel_cost / duration_days
        target_nightly = (budget * _HOTEL_BUDGET_SHARE) / duration_days
        if current_nightly > target_nightly:
            suggestions.append(
                f"Find a hotel under ${target_nightly:.0f}/night "
                f"(current cheapest: ${current_nightly:.0f}/night)"
            )

    # Flight suggestion
    if bd.flight_cost:
        target_flight = budget * _FLIGHT_BUDGET_SHARE
        if bd.flight_cost > target_flight:
            suggestions.append(
                f"Look for flights under ${target_flight:.0f} total "
                f"(current: ${bd.flight_cost:.0f})"
            )

    # Duration reduction suggestion
    if duration_days and duration_days > 3 and overage > 0:
        daily_rate = bd.total_estimated / duration_days if bd.total_estimated else 0
        days_to_cut = max(1, int(overage / daily_rate)) if daily_rate else 1
        suggestions.append(
            f"Reduce trip by {days_to_cut} day(s) to bring total under ${budget:.0f}"
        )

    # Activities suggestion
    if bd.activities_cost and bd.activities_cost > budget * _ACTIVITIES_BUDGET_SHARE:
        target = budget * _ACTIVITIES_BUDGET_SHARE
        suggestions.append(
            f"Limit paid activities to ${target:.0f} total "
            f"(current: ${bd.activities_cost:.0f}) — include free options"
        )

    return suggestions


def _completeness_suggestions(completeness: dict) -> List[str]:
    """Returns one specific suggestion per missing plan section."""
    labels = {
        "has_flights":        "Add at least one flight option with price and airline",
        "has_hotels":         "Add at least one hotel option with price per night",
        "has_activities":     "Add recommended activities with estimated costs",
        "has_visa_info":      "Include visa requirements for the traveler's passport country",
        "has_cost_breakdown": "Add a total cost breakdown (flights + hotel + activities)",
    }
    return [
        labels[key]
        for key, present in completeness.items()
        if not present and key in labels
    ]


# ── Main public function ───────────────────────────────────────────────────────

def critique_plan(state: Dict[str, Any]) -> CritiqueResult:
    """
    Critiques the travel plan produced by the master planner.

    Pure function — reads from state, returns CritiqueResult, touches nothing else.
    Student 4 calls this from a graph node and stores the result in state.

    Returns CritiqueResult with:
      passed      — False if budget is exceeded or required sections are missing
      score       — 0–10 quality score
      reason      — one-line summary
      issues      — specific problems found
      suggestions — actionable fixes for the planner
      budget      — BudgetBreakdown with all cost figures
      completeness — which sections are present in the plan
    """
    # ── 1. Extract inputs from state ──────────────────────────────────────────
    trip_ctx   = state.get("trip_context") or {}
    structured = state.get("planner_structured_results") or {}
    raw_results = state.get("planner_task_results") or {}

    budget = state.get("total_budget")
    if budget is None:
        budget = trip_ctx.get("total_budget")
    if budget is not None:
        budget = float(budget)

    duration_days = trip_ctx.get("duration_days")
    num_travelers = int(trip_ctx.get("num_travelers") or state.get("num_travelers") or 1)

    # ── 2. Completeness check ─────────────────────────────────────────────────
    flights    = structured.get("flights", [])
    hotels     = structured.get("hotels", [])
    activities = structured.get("activities", [])
    visa       = structured.get("visa")
    cost_obj   = structured.get("cost") or {}

    # Live flights (SerpAPI) are stored in raw_results under "fetch_live_flights".
    # Treat them as equivalent to DB flights for completeness and budget purposes.
    import json as _json
    live_flights_raw = raw_results.get("fetch_live_flights", "")
    live_flights: list = []
    if live_flights_raw and not flights:
        try:
            live_flights = _json.loads(live_flights_raw)
        except (ValueError, TypeError):
            live_flights = []
    has_flights = bool(flights) or bool(live_flights)

    completeness = {
        "has_flights":        has_flights,
        "has_hotels":         bool(hotels),
        "has_activities":     bool(activities),
        "has_visa_info":      bool(visa),
        "has_cost_breakdown": bool(cost_obj.get("total_cost")),
    }

    # ── 3. Budget breakdown ───────────────────────────────────────────────────
    # Inject live flights into structured so _build_budget_breakdown can price them.
    structured_for_budget = dict(structured)
    if live_flights and not structured_for_budget.get("flights"):
        # Normalise live flights to the same shape as DB flights
        structured_for_budget["flights"] = [
            {"price": f.get("price", 0)} for f in live_flights
        ]
    bd = _build_budget_breakdown(structured_for_budget, budget, duration_days, num_travelers)

    # Raw text fallback if structured results have no total
    if not bd.total_estimated and raw_results:
        raw_text = " ".join(str(v) for v in raw_results.values())
        bd.total_estimated = _extract_cost_from_raw_text(raw_text)
        if bd.total_estimated and budget:
            bd.overage = bd.total_estimated - budget
            bd.within_budget = bd.total_estimated <= budget

    # ── 4. Collect issues ─────────────────────────────────────────────────────
    issues: List[str] = []

    # Hard fail: budget exceeded
    budget_passed = True
    if bd.total_estimated and bd.budget and not bd.within_budget:
        budget_passed = False
        issues.append(
            f"Estimated cost ${bd.total_estimated:.0f} exceeds budget "
            f"${bd.budget:.0f} by ${bd.overage:.0f}"
        )

    # Soft fail: missing plan sections
    missing_sections = [k for k, present in completeness.items() if not present]
    for key in missing_sections:
        label = key.replace("has_", "").replace("_", " ")
        issues.append(f"Plan is missing: {label}")

    # ── 5. Build suggestions ──────────────────────────────────────────────────
    suggestions: List[str] = []
    suggestions.extend(_budget_suggestions(bd, duration_days))
    suggestions.extend(_completeness_suggestions(completeness))

    # ── 6. Score ──────────────────────────────────────────────────────────────
    score = 10
    if not budget_passed:
        score -= _BUDGET_OVERAGE_PENALTY
    score -= len(missing_sections) * _SCORE_PENALTY_PER_ISSUE
    score = max(0, min(10, score))

    # ── 7. Passed / reason ────────────────────────────────────────────────────
    # Hard fail only on budget. Missing sections reduce score but don't block.
    passed = budget_passed

    if passed and not issues:
        reason = (
            f"Plan is within budget (${bd.total_estimated:.0f} / ${bd.budget:.0f})"
            if bd.total_estimated and bd.budget
            else "Plan passed all checks."
        )
    elif not budget_passed:
        reason = (
            f"Budget exceeded by ${bd.overage:.0f} — "
            f"plan costs ${bd.total_estimated:.0f}, budget is ${bd.budget:.0f}"
        )
    else:
        reason = f"Plan passed budget check but has {len(missing_sections)} incomplete section(s)."

    logger.info(
        "Critic: passed=%s score=%d issues=%d suggestions=%d",
        passed, score, len(issues), len(suggestions),
    )

    return CritiqueResult(
        passed=passed,
        score=score,
        reason=reason,
        issues=issues,
        suggestions=suggestions,
        budget=bd,
        completeness=completeness,
    )
