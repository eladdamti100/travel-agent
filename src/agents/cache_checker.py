"""
Cache checker agent.

This agent runs after the master orchestrator routes a request to cache_check.

It checks whether the current user query is semantically similar enough to a
previously cached full-trip answer.

If cache hit:
    - writes cache_status="hit"
    - adds the cached answer as an AIMessage
    - router will end the graph

If cache miss:
    - writes cache_status="miss"
    - router will continue to the master planner
"""

from langchain_core.messages import AIMessage

from src.graph.state import AgentState
from src.models.cache import CacheStatus
from src.services.semantic_cache import (
    DEFAULT_HIT_THRESHOLD,
    build_trip_cache_key,
    find_cached_answer,
)
from src.utils.logger import get_logger

from src.agents.context_enricher import extract_trip_context_deterministic


logger = get_logger("cache_checker")


def run_cache_check(state: AgentState) -> dict:
    """
    Checks semantic cache for the latest user query.

    This function does not call the planner.
    It only decides whether a previous answer can be reused.

    If force_replan is True (user is modifying trip parameters), bypass cache.
    """
    messages = state.get("messages", [])
    force_replan = state.get("force_replan", False)

    if not messages:
        return {
            "cache_status": CacheStatus.MISS.value,
            "cache_similarity_score": 0.0,
            "cache_matched_query": "",
            "cache_answer": "",
        }

    # If user is modifying parameters, force re-planning
    if force_replan:
        logger.info("Cache checker: force_replan=True, bypassing cache to re-plan with modifications")
        return {
            "cache_status": CacheStatus.MISS.value,
            "cache_similarity_score": 0.0,
            "cache_matched_query": "",
            "cache_answer": "",
        }

    # --- START OF MINIMAL CHANGE: Canonical Trip Context Key ---
    ctx = extract_trip_context_deterministic(state)
    query = build_trip_cache_key(
        origin_airport=ctx.origin_airport,
        origin_country=ctx.origin_country,
        destination_city=ctx.destination_city,
        duration_days=ctx.duration_days,
        total_budget=ctx.total_budget,
    ) or getattr(messages[-1], "content", "")

    logger.info("Cache checker TripContext: %s", ctx.model_dump())
    logger.info("Cache checker query/key: %s", query)

    result = find_cached_answer(
        query=query,
        route="cache_check",
        threshold=DEFAULT_HIT_THRESHOLD,
    )

    logger.info(
    "Cache checker result: status=%s score=%.4f matched=%s reason=%s",
    result.status.value,
    result.similarity_score,
    result.matched_query,
    result.reason,
    )

    if result.status == CacheStatus.HIT:
        logger.info(
            "Cache checker hit. score=%.4f matched_query=%s",
            result.similarity_score,
            result.matched_query,
        )

        return {
            "cache_status": CacheStatus.HIT.value,
            "cache_similarity_score": result.similarity_score,
            "cache_matched_query": result.matched_query or "",
            "cache_answer": result.cached_answer or "",
            "messages": [AIMessage(content=result.cached_answer or "")],
        }

    logger.info(
        "Cache checker miss. score=%.4f matched_query=%s",
        result.similarity_score,
        result.matched_query,
    )

    return {
        "cache_status": CacheStatus.MISS.value,
        "cache_similarity_score": result.similarity_score,
        "cache_matched_query": result.matched_query or "",
        "cache_answer": "",
    }