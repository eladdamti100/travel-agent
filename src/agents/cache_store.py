"""
Cache store agent.

Stores successful full-trip answers in the semantic cache after a cache miss.

This is used only for requests that passed through cache_check and then
continued to the legacy planner/agent because no cached answer was found.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.graph.state import AgentState
from src.models.cache import CacheStatus
from src.services.semantic_cache import store_cache_entry
from src.utils.logger import get_logger

logger = get_logger("cache_store")


def run_cache_store(state: AgentState) -> dict:
    """
    Stores the latest final AI answer in semantic cache when appropriate.

    Store only when:
      - cache_status == "miss"
      - there is a latest user query
      - there is a latest final AI message
      - the final AI message is not a tool-call message
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

    try:
        store_cache_entry(
            query=query,
            answer=answer,
            route="cache_check",
        )
        logger.info("Cache store saved answer for query=%s", query)
    except Exception as error:
        logger.error("Cache store failed: %s", error)

    return {}


def _get_latest_user_query(state: AgentState) -> str:
    """
    Returns the latest HumanMessage content from graph state.
    """
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