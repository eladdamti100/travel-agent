"""
Tests for cache store — run_cache_store
Verifies that HITL questions, tool-call messages, and cache hits are never stored.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="Plan a trip to Paris for 5 days")],
        "cache_status": None,
        "awaiting_user_clarification": False,
    }
    base.update(overrides)
    return base


def _sync_executor_side_effect(fn, *args, **kwargs):
    """Run submitted functions synchronously so assertions are deterministic."""
    fn(*args, **kwargs)
    return MagicMock()


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

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor_side_effect
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

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec, \
             patch("src.agents.cache_store.compress_answer", return_value="compressed"):
            mock_exec.submit.side_effect = _sync_executor_side_effect
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

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor_side_effect
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

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor_side_effect
            run_cache_store(state)
            mock_store.assert_not_called()

    def test_missing_query_skips_store(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            messages=[
                AIMessage(content="Here is your Paris trip plan!"),
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor_side_effect
            run_cache_store(state)
            mock_store.assert_not_called()

    def test_missing_answer_skips_store(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            messages=[
                HumanMessage(content="Plan a trip to Paris"),
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor_side_effect
            run_cache_store(state)
            mock_store.assert_not_called()

    def test_none_cache_status_skips_store(self):
        from src.agents.cache_store import run_cache_store

        state = _make_state(
            cache_status=None,
            messages=[
                HumanMessage(content="Plan a trip to Paris"),
                AIMessage(content="Here is your trip plan!"),
            ],
        )

        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor_side_effect
            run_cache_store(state)
            mock_store.assert_not_called()
