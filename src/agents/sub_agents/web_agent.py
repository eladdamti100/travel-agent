"""
WebAgent — async multi-API intelligence layer for the master planner.

Load-balancing rule: if the number of tasks exceeds _MAX_TASKS_PER_INSTANCE,
the task list is split into two equal halves and both halves are executed as
separate concurrent "instances" via asyncio.gather. The extra instance is
ephemeral — it is created, awaited, and garbage-collected inside run().
"""

import asyncio
import json
from typing import Dict, Optional, Tuple

from src.agents.sub_agents.base import BaseSubAgent
from src.models.planner import PlannerTaskType, PlannerToolResults
from src.models.trip_context import TripContext
from src.tools.web_api_tools import (
    fetch_country_metadata,
    fetch_live_events,
    fetch_local_breweries,
    geocode_location,
    live_currency_conversion,
    web_research_tavily,
)
from src.utils.logger import get_logger

logger = get_logger("web_agent")

# Destination city → ISO-4217 local currency
_CITY_CURRENCY: Dict[str, str] = {
    "london":   "GBP",
    "paris":    "EUR",
    "berlin":   "EUR",
    "tokyo":    "JPY",
    "new york": "USD",
}

# Origin/passport country → ISO-4217 currency
_ORIGIN_CURRENCY: Dict[str, str] = {
    "israel":         "ILS",
    "united states":  "USD",
    "usa":            "USD",
    "united kingdom": "GBP",
    "uk":             "GBP",
    "france":         "EUR",
    "germany":        "EUR",
    "japan":          "JPY",
    "australia":      "AUD",
    "canada":         "CAD",
    "india":          "INR",
}

# Spawn a second parallel instance when task count exceeds this threshold
_MAX_TASKS_PER_INSTANCE: int = 3

# Internal sentinel key for the passport-currency sub-task (never exposed as a result key)
_PASSPORT_KEY = "__passport_currency__"


class WebAgent(BaseSubAgent):
    """
    Sub-agent executing live internet intelligence missions concurrently.

    When the task count exceeds _MAX_TASKS_PER_INSTANCE, a second virtual
    instance is spawned for the overflow batch and destroyed after merging.
    """

    agent_name = "web_agent"

    result_keys: Tuple[str, ...] = (
        PlannerTaskType.GEOCODE_LOCATION.value,
        PlannerTaskType.FETCH_LIVE_EVENTS.value,
        PlannerTaskType.LIVE_CURRENCY_CONVERSION.value,
        PlannerTaskType.FETCH_BREWERIES.value,
        PlannerTaskType.FETCH_COUNTRY_METADATA.value,
        PlannerTaskType.WEB_RESEARCH_TAVILY.value,
    )

    async def run(self, *, context: TripContext) -> PlannerToolResults:
        target_city    = context.destination_city    or "London"
        target_country = context.destination_country or "United Kingdom"
        dest_currency  = _CITY_CURRENCY.get(target_city.lower(), "EUR")
        origin_currency = _ORIGIN_CURRENCY.get(
            (context.origin_country or "").lower(), "USD"
        )
        budget = float(context.total_budget or 1000.0)

        tasks: Dict[str, object] = {
            PlannerTaskType.GEOCODE_LOCATION.value: geocode_location.ainvoke(
                {"city": target_city}
            ),
            PlannerTaskType.FETCH_LIVE_EVENTS.value: fetch_live_events.ainvoke(
                {"city": target_city}
            ),
            PlannerTaskType.FETCH_BREWERIES.value: fetch_local_breweries.ainvoke(
                {"city": target_city}
            ),
            PlannerTaskType.FETCH_COUNTRY_METADATA.value: fetch_country_metadata.ainvoke(
                {"country_name": target_country}
            ),
            PlannerTaskType.WEB_RESEARCH_TAVILY.value: web_research_tavily.ainvoke({
                "query": (
                    f"Top traveler alerts and seasonal cultural tips "
                    f"for {target_city} {context.travel_month or ''}"
                )
            }),
            PlannerTaskType.LIVE_CURRENCY_CONVERSION.value: live_currency_conversion.ainvoke({
                "amount": budget,
                "base": "USD",
                "target": dest_currency,
            }),
            _PASSPORT_KEY: live_currency_conversion.ainvoke({
                "amount": 1.0,
                "base": dest_currency,
                "target": origin_currency,
            }),
        }

        # ── Load balancing ────────────────────────────────────────────────────
        items = list(tasks.items())
        if len(items) > _MAX_TASKS_PER_INSTANCE:
            batch1 = dict(items[:_MAX_TASKS_PER_INSTANCE])
            batch2 = dict(items[_MAX_TASKS_PER_INSTANCE:])
            logger.info(
                "WebAgent overflow: %d tasks > threshold %d "
                "— spawning second instance for %d tasks",
                len(items), _MAX_TASKS_PER_INSTANCE, len(batch2),
            )
            raw1, raw2 = await asyncio.gather(
                self._execute_batch(batch1),
                self._execute_batch(batch2),
            )
            raw: Dict[str, str] = {**raw1, **raw2}
        else:
            raw = await self._execute_batch(tasks)
        # ─────────────────────────────────────────────────────────────────────

        # Enrich the currency result with inverse rate + passport conversion
        currency_raw = raw.pop(PlannerTaskType.LIVE_CURRENCY_CONVERSION.value, None)
        passport_raw = raw.pop(_PASSPORT_KEY, None)
        if currency_raw:
            raw[PlannerTaskType.LIVE_CURRENCY_CONVERSION.value] = self._enrich_currency(
                currency_raw=currency_raw,
                passport_raw=passport_raw,
                dest_currency=dest_currency,
                origin_currency=origin_currency,
                budget=budget,
            )

        result = PlannerToolResults()
        result.raw_results.update(raw)
        logger.info("WebAgent completed tasks: %s", list(result.raw_results.keys()))
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _execute_batch(self, tasks: Dict) -> Dict[str, str]:
        """Runs a dict of coroutines concurrently; returns string results."""
        keys   = list(tasks.keys())
        values = await asyncio.gather(*tasks.values(), return_exceptions=True)
        out: Dict[str, str] = {}
        for key, val in zip(keys, values):
            if isinstance(val, Exception):
                logger.error("WebAgent task failed. key=%s error=%s", key, val)
            else:
                out[str(key)] = str(val)
        return out

    @staticmethod
    def _enrich_currency(
        currency_raw: str,
        passport_raw: Optional[str],
        dest_currency: str,
        origin_currency: str,
        budget: float,
    ) -> str:
        """Adds inverse_rate, destination_currency, and passport conversion to the result."""
        try:
            base = json.loads(currency_raw)
        except (json.JSONDecodeError, TypeError):
            return currency_raw

        rate = base.get("rate", 1.0)
        try:
            inverse = round(1.0 / float(rate), 4) if rate else 0.0
        except (ZeroDivisionError, ValueError):
            inverse = 0.0

        base["inverse_rate"]       = inverse
        base["destination_currency"] = dest_currency
        base["passport_currency"]  = origin_currency

        if passport_raw:
            try:
                p = json.loads(passport_raw)
                p_rate = p.get("rate")
                if p_rate:
                    # 1 dest-currency in passport-currency
                    base["one_dest_in_passport"] = p.get("converted", "—")
                    # Full budget in passport-currency
                    passport_amount = budget * float(rate) * float(p_rate)
                    base["budget_in_passport"] = (
                        f"{passport_amount:,.2f} {origin_currency}"
                    )
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        return json.dumps(base)
