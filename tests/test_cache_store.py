"""
Tests for cache store — run_cache_store
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


def _sync_executor(fn, *args, **kwargs):
    fn(*args, **kwargs)
    return MagicMock()


class TestCacheStore:

    def test_skips_when_not_eligible(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        msgs = [HumanMessage(content="Plan"), AIMessage(content="Answer")]
        cases = [
            _make_state(awaiting_user_clarification=True, cache_status=CacheStatus.MISS.value, messages=msgs),
            _make_state(cache_status=CacheStatus.HIT.value, messages=msgs),
            _make_state(cache_status=None, messages=msgs),
        ]
        for state in cases:
            with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
                 patch("src.agents.cache_store._executor") as mock_exec:
                mock_exec.submit.side_effect = _sync_executor
                run_cache_store(state)
                mock_store.assert_not_called()

    def test_final_answer_saved_after_miss(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            # trip_context must carry destination_city or run_cache_store exits early.
            trip_context={"destination_city": "Paris"},
            messages=[
                HumanMessage(content="Plan a trip to Paris"),
                AIMessage(content="Here is your complete Paris trip plan!"),
            ],
        )
        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec, \
             patch("src.agents.cache_store.compress_answer", return_value="compressed"):
            mock_exec.submit.side_effect = _sync_executor
            run_cache_store(state)
        mock_store.assert_called_once()
        kw = mock_store.call_args.kwargs
        assert kw["query"] == "Plan a trip to Paris"
        assert "Paris trip plan" in kw["answer"]

    def test_tool_call_message_not_cached(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        state = _make_state(
            cache_status=CacheStatus.MISS.value,
            messages=[
                HumanMessage(content="Paris trip"),
                AIMessage(content="", tool_calls=[{"name": "fetch_flights", "args": {}, "id": "1"}]),
            ],
        )
        with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_exec.submit.side_effect = _sync_executor
            run_cache_store(state)
        mock_store.assert_not_called()

    def test_incomplete_state_skips_store(self):
        from src.agents.cache_store import run_cache_store
        from src.models.cache import CacheStatus

        # No human message (query missing)
        no_query = _make_state(cache_status=CacheStatus.MISS.value, messages=[AIMessage(content="Plan!")])
        # No AI message (answer missing)
        no_answer = _make_state(cache_status=CacheStatus.MISS.value, messages=[HumanMessage(content="Plan a trip")])

        for state in [no_query, no_answer]:
            with patch("src.agents.cache_store.store_cache_entry") as mock_store, \
                 patch("src.agents.cache_store._executor") as mock_exec:
                mock_exec.submit.side_effect = _sync_executor
                run_cache_store(state)
                mock_store.assert_not_called()
