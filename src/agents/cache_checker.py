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

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState
from src.models.cache import CacheStatus
from src.services.semantic_cache import (
    DEFAULT_HIT_THRESHOLD,
    find_cached_answer,
)
from src.utils.logger import get_logger

from src.agents.context_enricher import extract_trip_context_deterministic


logger = get_logger("cache_checker")


def run_cache_check(state: AgentState) -> dict:
    """
    Checks semantic cache for the latest user query.

    Three-layer lookup (all inside find_cached_answer):
      1. SQLite pre-filter  — destination exact match + budget ±5%
      2. Python hard filter — duration bucket
      3. Cosine similarity  — best semantic match above threshold

    Returns HIT (reuse cached answer) or MISS (continue to planner).
    Bypasses cache when force_replan=True.
    """
    messages = state.get("messages", [])
    force_replan = state.get("force_replan", False)

    raw_text = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    if not messages or not raw_text:
        return {
            "cache_status": CacheStatus.MISS.value,
            "cache_similarity_score": 0.0,
            "cache_matched_query": "",
            "cache_answer": "",
            "planning_query": "",
        }

    if force_replan:
        logger.info("Cache checker: force_replan=True, bypassing cache.")
        return {
            "cache_status": CacheStatus.MISS.value,
            "cache_similarity_score": 0.0,
            "cache_matched_query": "",
            "cache_answer": "",
            "planning_query": raw_text,
        }

    ctx = extract_trip_context_deterministic(state)
    logger.info("Cache checker TripContext: %s", ctx.model_dump())

    result = find_cached_answer(
        query=raw_text,
        route="cache_check",
        threshold=DEFAULT_HIT_THRESHOLD,
        trip_context=ctx.model_dump(),
    )

    query = raw_text
    # Save the original planning query before HITL adds more messages to state.
    # cache_store reads this to ensure store and lookup use the same text.
    planning_query = raw_text

    logger.info(
        "Cache checker result: status=%s score=%.4f matched=%s reason=%s",
        result.status.value,
        result.similarity_score,
        result.matched_query,
        result.reason,
    )

    if result.status == CacheStatus.HIT:
        logger.info(
            "Cache checker hit. score=%.4f matched_query=%s source=%s ttl_days=%s",
            result.similarity_score,
            result.matched_query,
            result.source,
            result.ttl_days,
        )

        return {
            "cache_status": CacheStatus.HIT.value,
            "cache_similarity_score": result.similarity_score,
            "cache_matched_query": result.matched_query or "",
            "cache_answer": result.cached_answer or "",
            "planning_query": planning_query,
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
        "planning_query": planning_query,
    }