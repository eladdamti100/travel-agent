"""
Tests for cache store — run_cache_store
Verifies that HITL questions, tool-call messages, and cache hits are never stored.
"""

from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="Plan a trip to Paris for 5 days")],
        "cache_status": None,
        "awaiting_user_clarification": False,
    }
    base.update(overrides)
    return base


def _should_skip_caching(state: dict) -> bool:
    """Check if caching should be skipped based on state conditions."""
    # Skip if awaiting user clarification
    if state.get("awaiting_user_clarification", False):
        return True
    
    # Skip if cache hit
    from src.models.cache import CacheStatus
    if state.get("cache_status") == CacheStatus.HIT.value:
        return True
    
    # Skip if last message is a tool call
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return True
    
    return False


class TestCacheStore:

    def test_hitl_question_not_stored_when_awaiting_clarification(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            awaiting_user_clarification=True,
            messages=[
                HumanMessage(content="Plan a trip to Paris"),
                AIMessage(content="Which airport are you flying from?"),
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store:
            run_cache_store(state)
            mock_store.assert_not_called()

    def test_final_answer_saved_after_miss(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            messages=[
                HumanMessage(content="Plan a trip to Paris"),
                AIMessage(content="Here is your complete Paris trip plan!"),
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store:
            run_cache_store(state)
            mock_store.assert_called_once()
            call_kwargs = mock_store.call_args.kwargs
            assert call_kwargs["query"] == "Plan a trip to Paris"
            assert "Paris trip plan" in call_kwargs["answer"]

    def test_cache_store_skips_on_cache_hit(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.HIT.value,
            messages=[
                HumanMessage(content="Trip to Paris"),
                AIMessage(content="Cached answer here"),
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store:
            run_cache_store(state)
            mock_store.assert_not_called()

    def test_tool_call_message_not_cached(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        tool_call_message = AIMessage(
            content="",
            tool_calls=[{"name": "fetch_flights", "args": {}, "id": "1"}],
        )
        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            messages=[
                HumanMessage(content="Paris trip"),
                tool_call_message,
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store:
            run_cache_store(state)
            mock_store.assert_not_called()