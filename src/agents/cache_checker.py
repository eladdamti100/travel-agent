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
    - router will continue to the legacy planner agent for now
"""

from langchain_core.messages import AIMessage

from src.graph.state import AgentState
from src.models.cache import CacheStatus
from src.services.semantic_cache import DEFAULT_HIT_THRESHOLD, find_cached_answer
from src.utils.logger import get_logger

logger = get_logger("cache_checker")


def run_cache_check(state: AgentState) -> dict:
    """
    Checks semantic cache for the latest user query.

    This function does not call the planner.
    It only decides whether a previous answer can be reused.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "cache_status": CacheStatus.MISS.value,
            "cache_similarity_score": 0.0,
            "cache_matched_query": "",
            "cache_answer": "",
        }

    query = getattr(messages[-1], "content", "")

    result = find_cached_answer(
        query=query,
        route="cache_check",
        threshold=DEFAULT_HIT_THRESHOLD,
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