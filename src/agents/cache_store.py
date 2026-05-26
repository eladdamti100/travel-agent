"""
Cache store agent.

Stores successful full-trip answers in the semantic cache after a cache miss.

This is used only for requests that passed through cache_check and then
continued to the legacy planner/agent because no cached answer was found.
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState
from src.models.cache import CacheStatus
from src.services.cache_compression import compress_answer
from src.services.semantic_cache import store_cache_entry
from src.utils.logger import get_logger

logger = get_logger("cache_store")

_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="cache_store")


def _background_store(query: str, answer: str) -> None:
    """
    Compresses and stores a cache entry in the background.

    Runs in a daemon thread — failures are logged but never propagated,
    so the user experience is never affected.
    """
    try:
        compressed = compress_answer(answer)
        store_cache_entry(
            query=query,
            answer=answer,
            route="cache_check",
            compressed_answer=compressed,
        )
        logger.info("Background cache store completed for query=%s", query)
    except Exception as error:
        logger.error("Background cache store failed: %s", error)


def run_cache_store(state: AgentState) -> dict:
    """
    Fires a background thread to store the latest final AI answer in the
    semantic cache, then returns immediately without blocking the user.

    Skips when:
      - awaiting_user_clarification is True (HITL mid-conversation)
      - cache_status != "miss"
      - query or final answer is missing
    """
    if state.get("awaiting_user_clarification"):
        logger.info("Cache store skipped: awaiting user clarification (HITL).")
        return {}

    if state.get("cache_status") != CacheStatus.MISS.value:
        return {}

    query = _get_latest_user_query(state)
    answer = _get_latest_final_ai_answer(state)

    if not query or not answer:
        logger.info("Cache store skipped: missing query or final answer.")
        return {}

    _executor.submit(_background_store, query, answer)
    logger.info("Cache store submitted to thread pool for query=%s", query)

    return {}


def _get_latest_user_query(state: AgentState) -> str:
    """
    Returns the latest HumanMessage content from graph state or its canonical structured string.
    """
    # --- START OF MINIMAL CHANGE: Canonical Trip Context Key ---
    from src.agents.context_enricher import extract_trip_context_deterministic
    ctx = extract_trip_context_deterministic(state)

    if ctx.destination_city and ctx.duration_days and ctx.origin_airport:
        return (
            f"origin_airport: {ctx.origin_airport} | "
            f"origin_country: {ctx.origin_country or 'unknown'} | "
            f"destination_city: {ctx.destination_city} | "
            f"duration_days: {ctx.duration_days} | "
            f"total_budget: {int(ctx.total_budget) if ctx.total_budget else 0}"
        )
    # --- END OF MINIMAL CHANGE ---

    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)

    return ""


def _get_latest_final_ai_answer(state: AgentState) -> str:
    """
    Returns the latest final AIMessage content that is not a tool-call request.
    """
    for message in reversed(state.get("messages", [])):
        if not isinstance(message, AIMessage):
            continue

        if getattr(message, "tool_calls", None):
            continue

        content = message.content
        return content if isinstance(content, str) else str(content)

    return ""