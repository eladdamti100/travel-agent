"""
Tests for cache hit/miss — run_cache_check
"""

import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(**overrides) -> dict:
    """
    Helper function to create a default state dictionary for cache check tests,
    with optional overrides for specific fields.
    """
    base = {
        "messages": [HumanMessage(content="Plan a trip to Paris for 5 days")],
        "cache_status": None,
        "cache_similarity_score": 0.0,
        "cache_matched_query": None,
        "cache_answer": None,
    }
    base.update(overrides)
    return base


def test_cache_check_hit():
    from src.agents.cache_checker import run_cache_check
    from src.models.cache import CacheStatus
    import src.agents.cache_checker as cache_checker_mod
    from unittest.mock import MagicMock

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


def test_cache_check_miss():
    from src.agents.cache_checker import run_cache_check
    from src.models.cache import CacheStatus
    import src.agents.cache_checker as cache_checker_mod
    from unittest.mock import MagicMock

    miss_result = MagicMock()
    miss_result.status = CacheStatus.MISS
    miss_result.similarity_score = 0.40
    miss_result.matched_query = ""
    miss_result.cached_answer = None

    with patch("src.agents.cache_checker.find_cached_answer", return_value=miss_result):
        result = run_cache_check(_make_state())

    assert result["cache_status"] == CacheStatus.MISS.value
    assert result["cache_answer"] in ("", None)
    assert result.get("messages") == [HumanMessage(content="Plan a trip to Paris for 5 days")]


def test_cache_check_empty_messages_returns_miss():
    from src.agents.cache_checker import run_cache_check
    from src.models.cache import CacheStatus

    result = run_cache_check(_make_state(messages=[]))
    assert result["cache_status"] == CacheStatus.MISS.value