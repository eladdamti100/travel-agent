"""
Manager Web Agent — contextual intelligence layer for the travel planner.

Responsibilities (per Epic 3 decomposition):
  LIVE_CURRENCY_CONVERSION — budget in destination currency + inverse rate
  FETCH_COUNTRY_METADATA   — country region, currency code, and capital
  WEB_RESEARCH_TAVILY      — live traveler alerts and seasonal tips

This agent uses direct async tool calls (no LLM loop) because all three
tasks are fully parametric — the inputs are derived deterministically from
TripContext and do not benefit from LLM reasoning.  This mirrors the design
of the original WebAgent and keeps the manager path fast and predictable.

Currency enrichment logic is preserved from the original WebAgent:
  - Primary conversion:  budget USD → destination currency
  - Inverse conversion:  destination currency → passport currency (1-unit)
  - Merged into a single enriched JSON blob for the plan formatter
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional, Tuple

from src.agents.sub_agents.base import BaseSubAgent
from src.config.city_registry import (
    CURRENCY_BY_CITY as _CITY_CURRENCY,
    CURRENCY_BY_COUNTRY as _ORIGIN_CURRENCY,
)
from src.models.planner import PlannerTaskType, PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.web_api_tools import (
    fetch_country_metadata,
    live_currency_conversion,
    web_research_tavily,
)
from src.utils.logger import get_logger

logger = get_logger("manager_web_agent")

# Internal sentinel — passport-rate sub-call; never written to raw_results
_PASSPORT_KEY = "__passport_currency__"


class ManagerWebAgent(BaseSubAgent):
    """
    Contextual intelligence agent: currency conversion, country metadata,
    and live general research.

    Replaces the three 'managerial' tasks that used to live in the monolithic
    WebAgent.  Direct async tool calls — no LLM — for speed and determinism.
    """

    agent_name = "manager_web_agent"
    result_keys: Tuple[str, ...] = (
        PlannerTaskType.LIVE_CURRENCY_CONVERSION.value,  # "live_currency_conversion"
        PlannerTaskType.FETCH_COUNTRY_METADATA.value,    # "fetch_country_metadata"
        PlannerTaskType.WEB_RESEARCH_TAVILY.value,       # "web_research_tavily"
    )

    async def run(self, *, context: TripContext) -> PlannerToolResults:
        result = PlannerToolResults()

        target_city    = context.destination_city
        target_country = context.destination_country or target_city

        if not target_city:
            logger.info("manager_web_agent. destination_city=None skipping=True")
            return result

        dest_currency   = _CITY_CURRENCY.get(target_city.lower(), "EUR")
        origin_currency = _ORIGIN_CURRENCY.get(
            (context.origin_country or "").lower(), "USD"
        )
        budget = float(context.total_budget or 1000.0)

        # ── Run all three tasks concurrently ─────────────────────────────────
        tasks = {
            PlannerTaskType.LIVE_CURRENCY_CONVERSION.value: live_currency_conversion.ainvoke({
                "amount": budget,
                "base":   "USD",
                "target": dest_currency,
            }),
            _PASSPORT_KEY: live_currency_conversion.ainvoke({
                "amount": 1.0,
                "base":   dest_currency,
                "target": origin_currency,
            }),
            PlannerTaskType.FETCH_COUNTRY_METADATA.value: fetch_country_metadata.ainvoke({
                "country_name": target_country,
            }),
            PlannerTaskType.WEB_RESEARCH_TAVILY.value: web_research_tavily.ainvoke({
                "query": (
                    f"{target_city} travel guide: local culture etiquette customs, "
                    f"metro transport tips getting around, safety alerts scams to avoid, "
                    f"must-see landmarks local food highlights"
                )
            }),
        }

        keys   = list(tasks.keys())
        values = await asyncio.gather(*tasks.values(), return_exceptions=True)
        raw    = {}
        for key, val in zip(keys, values):
            if isinstance(val, Exception):
                logger.error(
                    "manager_web_agent.task_failed. key=%s error=%s", key, val
                )
            else:
                raw[key] = str(val)

        # ── Currency enrichment (preserve original WebAgent logic) ────────────
        currency_raw: Optional[str] = raw.pop(PlannerTaskType.LIVE_CURRENCY_CONVERSION.value, None)
        passport_raw: Optional[str] = raw.pop(_PASSPORT_KEY, None)
        if currency_raw:
            raw[PlannerTaskType.LIVE_CURRENCY_CONVERSION.value] = _enrich_currency(
                currency_raw   = currency_raw,
                passport_raw   = passport_raw,
                dest_currency  = dest_currency,
                origin_currency = origin_currency,
                budget         = budget,
            )

        result.raw_results.update(raw)
        logger.info(
            "manager_web_agent.completed. destination=%s keys=%s",
            target_city, list(result.raw_results.keys()),
        )
        return result


# ── Currency enrichment helper (lifted from the original WebAgent) ────────────

def _enrich_currency(
    currency_raw:    str,
    passport_raw:    Optional[str],
    dest_currency:   str,
    origin_currency: str,
    budget:          float,
) -> str:
    """Adds inverse_rate, destination_currency, and passport conversion fields."""
    try:
        base = json.loads(currency_raw)
    except (json.JSONDecodeError, TypeError):
        return currency_raw

    rate = base.get("rate", 1.0)
    try:
        inverse = round(1.0 / float(rate), 4) if rate else 0.0
    except (ZeroDivisionError, ValueError):
        inverse = 0.0

    base["inverse_rate"]          = inverse
    base["destination_currency"]  = dest_currency
    base["passport_currency"]     = origin_currency

    if passport_raw:
        try:
            p       = json.loads(passport_raw)
            p_rate  = p.get("rate")
            if p_rate:
                base["one_dest_in_passport"] = p.get("converted", "—")
                passport_amount = budget * float(rate) * float(p_rate)
                base["budget_in_passport"] = (
                    f"{passport_amount:,.2f} {origin_currency}"
                )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    return json.dumps(base)
