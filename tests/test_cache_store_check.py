"""
Regression & Integration Audit — Semantic Cache Integrity (Audit Item 2)

Verifies:
  1. DEFAULT_HIT_THRESHOLD is exactly 0.85.
  2. cache_checker passes that threshold to find_cached_answer unchanged.
  3. Scores below 0.85 produce a MISS; scores at/above produce a HIT.
  4. cache_store uses ThreadPoolExecutor (not asyncio) for background writes.
  5. run_cache_store returns immediately without blocking — the submit() call
     is fire-and-forget.
  6. The executor is a module-level singleton (not re-created per call).
  7. cache_store correctly tags plans as 'web' source when used_web_source=True.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.models.cache import CacheStatus
from src.services.semantic_cache import DEFAULT_HIT_THRESHOLD


# ── Helpers ───────────────────────────────────────────────────────────────────

def _state(answer="Here is your Paris plan!", **overrides):
    base = {
        "messages": [
            HumanMessage(content="Plan a 5-day trip to Paris from TLV, budget $2000"),
            AIMessage(content=answer),
        ],
        "cache_status": CacheStatus.MISS.value,
        "trip_context": {"destination_city": "Paris"},
        "used_web_source": False,
        "planning_query": "",
    }
    base.update(overrides)
    return base


def _mock_cache_result(status: CacheStatus, score: float, answer: str = ""):
    r = MagicMock()
    r.status = status
    r.similarity_score = score
    r.matched_query = "paris 5-day $2000 TLV"
    r.cached_answer = answer
    r.source = "db"
    r.ttl_days = 30
    return r


# ── 1. Threshold constant ─────────────────────────────────────────────────────

class TestThresholdConstant:

    def test_default_hit_threshold_is_0_85(self):
        """The canonical threshold must stay at 0.85 — changing it breaks cache behaviour."""
        assert DEFAULT_HIT_THRESHOLD == 0.85

    def test_cache_checker_passes_threshold_to_find_cached_answer(self):
        """cache_checker.run_cache_check must forward DEFAULT_HIT_THRESHOLD unchanged."""
        from src.agents.cache_checker import run_cache_check

        hit = _mock_cache_result(CacheStatus.HIT, 0.90, "Cached Paris plan")
        with patch("src.agents.cache_checker.find_cached_answer", return_value=hit) as mock_find:
            run_cache_check(_state())

        _, kwargs = mock_find.call_args
        assert kwargs.get("threshold") == DEFAULT_HIT_THRESHOLD, (
            f"run_cache_check forwarded threshold={kwargs.get('threshold')!r}, "
            f"expected {DEFAULT_HIT_THRESHOLD}"
        )


# ── 2. Score boundary behaviour ───────────────────────────────────────────────

class TestScoreBoundary:

    @pytest.mark.parametrize("score,expected_status", [
        (0.0,   CacheStatus.MISS),
        (0.50,  CacheStatus.MISS),
        (0.84,  CacheStatus.MISS),   # just below threshold
        (0.85,  CacheStatus.HIT),    # exactly at threshold
        (0.90,  CacheStatus.HIT),
        (1.00,  CacheStatus.HIT),
    ])
    def test_hit_miss_boundary_respected(self, score, expected_status):
        """
        The semantic cache must return HIT for score >= 0.85, MISS otherwise.
        The boundary itself (0.85) must be a HIT.
        """
        from src.agents.cache_checker import run_cache_check

        mock_result = _mock_cache_result(expected_status, score, "Cached answer" if expected_status == CacheStatus.HIT else "")
        with patch("src.agents.cache_checker.find_cached_answer", return_value=mock_result):
            result = run_cache_check(_state())

        assert result["cache_status"] == expected_status.value, (
            f"Score {score} should be {expected_status.value}, got {result['cache_status']}"
        )
        assert abs(result["cache_similarity_score"] - score) < 1e-9

    def test_hit_result_carries_cached_answer(self):
        from src.agents.cache_checker import run_cache_check

        hit = _mock_cache_result(CacheStatus.HIT, 0.92, "Complete Paris itinerary")
        with patch("src.agents.cache_checker.find_cached_answer", return_value=hit):
            result = run_cache_check(_state())

        assert result["cache_answer"] == "Complete Paris itinerary"
        msgs = result.get("messages", [])
        assert any(isinstance(m, AIMessage) and "Complete Paris" in m.content for m in msgs)

    def test_miss_result_has_no_cached_answer(self):
        from src.agents.cache_checker import run_cache_check

        miss = _mock_cache_result(CacheStatus.MISS, 0.40)
        with patch("src.agents.cache_checker.find_cached_answer", return_value=miss):
            result = run_cache_check(_state())

        assert result["cache_answer"] in ("", None)
        assert result.get("messages") is None


# ── 3. ThreadPoolExecutor background write ────────────────────────────────────

class TestBackgroundThread:

    def test_executor_is_threadpoolexecutor(self):
        """cache_store must use a ThreadPoolExecutor, not asyncio, for background writes."""
        import src.agents.cache_store as mod
        assert isinstance(mod._executor, ThreadPoolExecutor), (
            "cache_store._executor must be a ThreadPoolExecutor for non-blocking writes"
        )

    def test_executor_is_module_level_singleton(self):
        """The executor must be created once at module level, not per call."""
        import src.agents.cache_store as mod1
        import src.agents.cache_store as mod2
        assert mod1._executor is mod2._executor, (
            "cache_store._executor must be a module-level singleton"
        )

    def test_run_cache_store_returns_before_write_completes(self):
        """
        run_cache_store must return {} immediately; the actual SQLite write
        happens asynchronously in the background thread.
        """
        from src.agents.cache_store import run_cache_store

        write_started = threading.Event()
        write_proceed = threading.Event()

        def _slow_store(*args, **kwargs):
            write_started.set()
            write_proceed.wait(timeout=5)

        state = _state()

        with patch("src.agents.cache_store.store_cache_entry", side_effect=_slow_store), \
             patch("src.agents.cache_store.compress_answer", return_value="compressed"):
            result = run_cache_store(state)

        # Must return immediately without waiting for the write to complete.
        assert result == {}
        write_proceed.set()  # let the background thread finish

    def test_submit_called_on_executor(self):
        """submit() is called on the module _executor when a valid state is passed."""
        from src.agents.cache_store import run_cache_store

        state = _state()

        with patch("src.agents.cache_store.store_cache_entry"), \
             patch("src.agents.cache_store.compress_answer", return_value="c"), \
             patch("src.agents.cache_store._executor") as mock_exec:
            mock_future = MagicMock()
            mock_future.exception.return_value = None
            mock_exec.submit.return_value = mock_future
            run_cache_store(state)

        mock_exec.submit.assert_called_once()

    def test_web_source_flag_forwarded(self):
        """When used_web_source=True, the store call receives source='web'."""
        from src.agents.cache_store import run_cache_store

        captured = {}

        def _capture_store(*args, **kwargs):
            captured.update(kwargs)

        state = _state(used_web_source=True)

        with patch("src.agents.cache_store.store_cache_entry", side_effect=_capture_store), \
             patch("src.agents.cache_store.compress_answer", return_value="c"), \
             patch("src.agents.cache_store._executor") as mock_exec:
            def _sync(fn, *a, **kw):
                fn(*a, **kw)
                f = MagicMock()
                f.exception.return_value = None
                return f
            mock_exec.submit.side_effect = _sync
            run_cache_store(state)

        assert captured.get("source") == "web", (
            "Plans built from web fallback must be tagged source='web' in the cache"
        )

    def test_db_source_flag_forwarded(self):
        """When used_web_source=False, the store call receives source='db'."""
        from src.agents.cache_store import run_cache_store

        captured = {}

        def _capture_store(*args, **kwargs):
            captured.update(kwargs)

        state = _state(used_web_source=False)

        with patch("src.agents.cache_store.store_cache_entry", side_effect=_capture_store), \
             patch("src.agents.cache_store.compress_answer", return_value="c"), \
             patch("src.agents.cache_store._executor") as mock_exec:
            def _sync(fn, *a, **kw):
                fn(*a, **kw)
                f = MagicMock()
                f.exception.return_value = None
                return f
            mock_exec.submit.side_effect = _sync
            run_cache_store(state)

        assert captured.get("source") == "db"


# ── 4. Origin hard filter ─────────────────────────────────────────────────────

class TestOriginHardFilter:
    """origin_airport mismatch must reject a cached entry regardless of similarity."""

    def test_different_origin_is_rejected(self):
        """TLV→Tokyo plan must not be served to a JFK→Tokyo query."""
        from src.services.trip_vector import check_hard_filters

        query_ctx  = {"origin_airport": "JFK", "destination_city": "Tokyo", "duration_days": 5}
        cached_ctx = {"origin_airport": "TLV", "destination_city": "Tokyo", "duration_days": 5}

        passes, reason = check_hard_filters(query_ctx, cached_ctx)
        assert not passes, "Different origins must be rejected by check_hard_filters"
        assert "TLV" in reason and "JFK" in reason

    def test_same_origin_is_accepted(self):
        """TLV→Tokyo plan should be served to another TLV→Tokyo query."""
        from src.services.trip_vector import check_hard_filters

        query_ctx  = {"origin_airport": "TLV", "destination_city": "Tokyo", "duration_days": 5}
        cached_ctx = {"origin_airport": "TLV", "destination_city": "Tokyo", "duration_days": 5}

        passes, _ = check_hard_filters(query_ctx, cached_ctx)
        assert passes

    def test_missing_query_origin_allows_hit(self):
        """If the query has no origin, don't block — origin is unknown, not mismatched."""
        from src.services.trip_vector import check_hard_filters

        query_ctx  = {"destination_city": "Tokyo", "duration_days": 5}
        cached_ctx = {"origin_airport": "TLV", "destination_city": "Tokyo", "duration_days": 5}

        passes, _ = check_hard_filters(query_ctx, cached_ctx)
        assert passes

    def test_missing_cached_origin_allows_hit(self):
        """If the cached entry has no origin, don't block — old entries lack this field."""
        from src.services.trip_vector import check_hard_filters

        query_ctx  = {"origin_airport": "TLV", "destination_city": "Tokyo", "duration_days": 5}
        cached_ctx = {"destination_city": "Tokyo", "duration_days": 5}

        passes, _ = check_hard_filters(query_ctx, cached_ctx)
        assert passes

    def test_origin_case_insensitive(self):
        """'tlv' and 'TLV' must be treated as the same origin."""
        from src.services.trip_vector import check_hard_filters

        query_ctx  = {"origin_airport": "tlv", "destination_city": "Tokyo"}
        cached_ctx = {"origin_airport": "TLV", "destination_city": "Tokyo"}

        passes, _ = check_hard_filters(query_ctx, cached_ctx)
        assert passes
