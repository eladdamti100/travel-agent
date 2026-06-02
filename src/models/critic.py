"""
Critic result models — structured output from the plan critique step.

Designed to be consumed by:
  - Student 4 (wires critic into the graph via nodes.py / workflow.py)
  - Student 5 (renders CritiqueResult in the UI)
  - The planner itself (suggestions feed back into a replan loop)
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BudgetBreakdown:
    """
    Itemised cost breakdown extracted from planner_structured_results.

    All monetary values are in the user's stated currency (default USD).
    None means the value could not be extracted from the plan.
    """

    budget: Optional[float] = None           # user's stated total budget
    flight_cost: Optional[float] = None      # cheapest available flight (per person)
    hotel_cost: Optional[float] = None       # price_per_night × duration_days
    activities_cost: Optional[float] = None  # sum of top activities
    total_estimated: Optional[float] = None  # flight + hotel + activities
    overage: Optional[float] = None          # total_estimated − budget (positive = over budget)
    within_budget: Optional[bool] = None     # True / False / None (unknown)


@dataclass
class CritiqueResult:
    """
    Structured output of the critic agent.

    passed        — False if any hard checks fail (budget exceeded, missing required sections)
    score         — 0–10 quality score for Student 5's UI display
    reason        — one-line summary of the outcome
    issues        — list of specific problems found in the plan
    suggestions   — actionable fixes specific enough for the planner to act on
                    e.g. "find a hotel under $90/night" not "reduce costs"
    budget        — itemised budget breakdown
    completeness  — which plan sections are present
    """

    passed: bool
    score: int                              # 0–10
    reason: str
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    budget: BudgetBreakdown = field(default_factory=BudgetBreakdown)
    completeness: dict = field(default_factory=dict)
    # {has_flights, has_hotels, has_activities, has_visa_info, has_cost_breakdown}
