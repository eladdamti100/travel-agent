"""
Tests for cache hit/miss — run_cache_check
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage(content="Plan a trip to Paris for 5 days, origin airport TLV, Israeli passport, budget $2000")],
        "cache_status": None,
        "cache_similarity_score": 0.0,
        "cache_matched_query": None,
        "cache_answer": None,
    }
    base.update(overrides)
    return base


class TestCacheCheck:

    def test_cache_hit(self):
        from src.agents.cache_checker import run_cache_check
        from src.models.cache import CacheStatus

        hit_result = MagicMock()
        hit_result.status = CacheStatus.HIT
        hit_result.similarity_score = 0.95
        hit_result.matched_query = "Paris trip"
        hit_result.cached_answer = "Here is your Paris trip plan."

        with patch("src.agents.cache_checker.find_cached_answer", return_value=hit_result):
            result = run_cache_check(_make_state())

        assert result["cache_status"] == CacheStatus.HIT.value
        assert result["cache_similarity_score"] == 0.95
        assert result["cache_matched_query"] == "Paris trip"
        assert result["cache_answer"] == "Here is your Paris trip plan."
        assert any(
            isinstance(m, AIMessage) and "Paris" in m.content
            for m in result.get("messages", [])
        )

    def test_cache_miss(self):
        from src.agents.cache_checker import run_cache_check
        from src.models.cache import CacheStatus

        miss_result = MagicMock()
        miss_result.status = CacheStatus.MISS
        miss_result.similarity_score = 0.40
        miss_result.matched_query = ""
        miss_result.cached_answer = None

        with patch("src.agents.cache_checker.find_cached_answer", return_value=miss_result):
            result = run_cache_check(_make_state())

        assert result["cache_status"] == CacheStatus.MISS.value
        assert result["cache_answer"] in ("", None)
        assert result.get("messages") is None

    def test_empty_messages_returns_miss(self):
        from src.agents.cache_checker import run_cache_check
        from src.models.cache import CacheStatus

        result = run_cache_check(_make_state(messages=[]))
        assert result["cache_status"] == CacheStatus.MISS.value

    def test_force_replan_bypasses_cache(self):
        from src.agents.cache_checker import run_cache_check
        from src.models.cache import CacheStatus

        hit_result = MagicMock()
        hit_result.status = CacheStatus.HIT
        hit_result.similarity_score = 0.95
        hit_result.matched_query = "Paris trip"
        hit_result.cached_answer = "Here is your Paris trip plan."

        with patch("src.agents.cache_checker.find_cached_answer", return_value=hit_result):
            # Even though cache would hit, force_replan=True should bypass it
            result = run_cache_check(_make_state(force_replan=True))

        # Should return MISS when force_replan is True
        assert result["cache_status"] == CacheStatus.MISS.value
        assert result["cache_answer"] == ""
        assert "messages" not in result or result.get("messages") is None
