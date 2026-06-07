"""
Tests for Mission 2 semantic cache changes:
  - valid_until column and expiry logic
  - _compute_valid_until
  - background sweep thread
  - normalize_query called once per find_cached_answer
  - WAL mode enabled
  - LIMIT enforced in _load_embedding_index
  - semantic HIT without pre-filter still returns answer (bug fix)
  - delete-by-id instead of delete-by-normalized-query (bug fix)
  - _extract_travel_start_date various formats
  - _compute_travel_end_date
  - store_cache_entry persists valid_until
"""

import json
import sqlite3
import threading
import time
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cache_db(tmp_path):
    """Redirects the module to an isolated tmp DB for the duration of the test."""
    import src.services.semantic_cache as mod
    orig_path, orig_init = mod._CACHE_DB_PATH, mod._db_initialized
    mod._CACHE_DB_PATH = tmp_path / "test_mission2.db"
    mod._db_initialized = False
    mod.initialize_cache_db()
    yield tmp_path / "test_mission2.db"
    mod._CACHE_DB_PATH = orig_path
    mod._db_initialized = orig_init


def _insert_row(db_path, *, query, source="db", ttl_days=30,
                valid_until=None, created_at=None, answer="answer text"):
    """Directly inserts a row into the test DB, bypassing the Python layer."""
    nq = query.lower().strip()
    ca = created_at or "datetime('now')"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO semantic_cache
              (query, normalized_query, answer, route, embedding_json,
               source, ttl_days, valid_until, created_at)
            VALUES (?, ?, ?, 'test', '[]', ?, ?, ?,
                    {ca if created_at is None else '?'})
            """,
            (query, nq, answer, source, ttl_days, valid_until)
            if created_at is None
            else (query, nq, answer, source, ttl_days, valid_until, created_at),
        )


# ── _compute_valid_until ──────────────────────────────────────────────────────

def test_compute_valid_until_start_date():
    from src.services.semantic_cache import _compute_valid_until
    ctx = {"travel_start_date": "2026-06-15"}
    assert _compute_valid_until(ctx) == "2026-06-14"


def test_compute_valid_until_end_date_only():
    from src.services.semantic_cache import _compute_valid_until
    ctx = {"travel_end_date": "2026-06-22"}
    assert _compute_valid_until(ctx) == "2026-06-22"


def test_compute_valid_until_start_takes_priority_over_end():
    from src.services.semantic_cache import _compute_valid_until
    ctx = {"travel_start_date": "2026-06-15", "travel_end_date": "2026-06-22"}
    # start_date - 1 day wins over end_date
    assert _compute_valid_until(ctx) == "2026-06-14"


def test_compute_valid_until_no_dates_returns_none():
    from src.services.semantic_cache import _compute_valid_until
    assert _compute_valid_until({}) is None
    assert _compute_valid_until(None) is None
    assert _compute_valid_until({"destination_city": "Paris"}) is None


def test_compute_valid_until_bad_date_string_returns_none():
    from src.services.semantic_cache import _compute_valid_until
    assert _compute_valid_until({"travel_start_date": "not-a-date"}) is None
    assert _compute_valid_until({"travel_end_date": "32/13/2026"}) is None


# ── valid_until DB filtering ──────────────────────────────────────────────────

def test_valid_until_past_is_invisible_to_lookup(cache_db):
    from src.services.semantic_cache import find_exact_cached_answer
    from src.models.cache import CacheStatus

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    _insert_row(cache_db, query="past trip query", valid_until=yesterday)

    result = find_exact_cached_answer("past trip query", route="test")
    assert result.status == CacheStatus.MISS


def test_valid_until_today_is_visible(cache_db):
    from src.services.semantic_cache import find_exact_cached_answer
    from src.models.cache import CacheStatus

    today = date.today().isoformat()
    _insert_row(cache_db, query="today trip query", valid_until=today)

    result = find_exact_cached_answer("today trip query", route="test")
    assert result.status == CacheStatus.HIT


def test_valid_until_null_falls_back_to_ttl(cache_db):
    from src.services.semantic_cache import find_exact_cached_answer
    from src.models.cache import CacheStatus

    # No valid_until; TTL=30 days; created yesterday → still valid
    _insert_row(
        cache_db,
        query="no date trip query",
        valid_until=None,
        created_at="datetime('now', '-1 days')",
    )
    result = find_exact_cached_answer("no date trip query", route="test")
    assert result.status == CacheStatus.HIT


# ── store_cache_entry persists valid_until ────────────────────────────────────

def test_store_persists_valid_until(cache_db):
    import src.services.semantic_cache as mod
    from src.services.semantic_cache import store_cache_entry

    ctx = {"travel_start_date": "2026-08-10", "destination_city": "Tokyo"}
    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        store_cache_entry("tokyo august trip", "plan", route="test",
                          trip_context=ctx)

    with sqlite3.connect(cache_db) as conn:
        row = conn.execute(
            "SELECT valid_until FROM semantic_cache WHERE normalized_query = ?",
            ("tokyo august trip",),
        ).fetchone()

    assert row is not None
    assert row[0] == "2026-08-09"  # start_date - 1 day


def test_store_valid_until_null_when_no_dates(cache_db):
    from src.services.semantic_cache import store_cache_entry

    ctx = {"destination_city": "London"}
    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        store_cache_entry("london trip", "plan", route="test", trip_context=ctx)

    with sqlite3.connect(cache_db) as conn:
        row = conn.execute(
            "SELECT valid_until FROM semantic_cache WHERE normalized_query = ?",
            ("london trip",),
        ).fetchone()

    assert row is not None
    assert row[0] is None


# ── Background sweep ──────────────────────────────────────────────────────────

def test_sweep_thread_is_daemon_and_alive():
    from src.services.semantic_cache import _sweep_thread
    assert _sweep_thread.is_alive()
    assert _sweep_thread.daemon


def test_sweep_expired_entries_removes_past_valid_until(cache_db):
    from src.services.semantic_cache import _sweep_expired_entries

    past = (date.today() - timedelta(days=1)).isoformat()
    _insert_row(cache_db, query="sweep me", valid_until=past)
    _insert_row(cache_db, query="keep me", valid_until=None)

    deleted = _sweep_expired_entries()
    assert deleted >= 1

    with sqlite3.connect(cache_db) as conn:
        remaining = [
            r[0] for r in conn.execute(
                "SELECT normalized_query FROM semantic_cache"
            ).fetchall()
        ]
    assert "sweep me" not in remaining
    assert "keep me" in remaining


def test_sweep_removes_ttl_expired_entries(cache_db):
    from src.services.semantic_cache import _sweep_expired_entries

    _insert_row(
        cache_db,
        query="old entry",
        ttl_days=1,
        created_at="2020-01-01T00:00:00",
    )
    deleted = _sweep_expired_entries()
    assert deleted >= 1


def test_sweep_on_missing_db_returns_zero(tmp_path):
    import src.services.semantic_cache as mod
    orig = mod._CACHE_DB_PATH
    mod._CACHE_DB_PATH = tmp_path / "nonexistent.db"
    try:
        assert mod._sweep_expired_entries() == 0
    finally:
        mod._CACHE_DB_PATH = orig


# ── normalize_query called once ───────────────────────────────────────────────

def test_normalize_query_called_once_per_find(cache_db):
    """find_cached_answer must normalize the query exactly once."""
    from src.services.semantic_cache import find_cached_answer, normalize_query

    call_count = {"n": 0}
    real_normalize = normalize_query

    def counting_normalize(q):
        call_count["n"] += 1
        return real_normalize(q)

    with patch("src.services.semantic_cache.normalize_query", side_effect=counting_normalize), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        find_cached_answer("paris trip 7 days", route="test")

    # find_exact_cached_answer reuses the pre-computed value; no second call.
    assert call_count["n"] == 1


def test_find_exact_accepts_precomputed_normalized(cache_db):
    from src.services.semantic_cache import find_exact_cached_answer, normalize_query
    from src.models.cache import CacheStatus

    nq = normalize_query("paris trip")
    # Should work without error whether or not the entry exists
    result = find_exact_cached_answer("paris trip", route="test", _normalized=nq)
    assert result.status == CacheStatus.MISS  # nothing stored — just no crash


# ── WAL mode ─────────────────────────────────────────────────────────────────

def test_wal_mode_enabled(cache_db):
    with sqlite3.connect(cache_db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


# ── LIMIT in _load_embedding_index ───────────────────────────────────────────

def test_load_embedding_index_respects_limit(cache_db):
    import src.services.semantic_cache as mod
    from src.services.semantic_cache import _load_embedding_index

    # Insert more rows than _MAX_ROWS_PER_ROUTE
    over = mod._MAX_ROWS_PER_ROUTE + 10
    with sqlite3.connect(cache_db) as conn:
        for i in range(over):
            conn.execute(
                "INSERT INTO semantic_cache "
                "(query, normalized_query, answer, route, embedding_json, "
                " source, ttl_days, created_at) "
                "VALUES (?, ?, 'ans', 'test', '[]', 'db', 30, datetime('now'))",
                (f"query {i}", f"query {i}"),
            )

    rows = _load_embedding_index("test")
    assert len(rows) <= mod._MAX_ROWS_PER_ROUTE


# ── Semantic HIT without pre-filter returns answer (bug fix) ──────────────────

def test_semantic_hit_no_prefilter_returns_answer(cache_db):
    """
    When no destination/budget in context, answer_cols is empty in _load_embedding_index.
    find_cached_answer must still return the answer by fetching it via _fetch_row_by_id.
    """
    from src.services.semantic_cache import find_cached_answer, store_cache_entry
    from src.models.cache import CacheStatus

    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        store_cache_entry("general travel query", "The full plan text", route="test")

    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        # No trip_context → no pre-filter → answer not in _load_embedding_index rows
        result = find_cached_answer(
            "general travel query",
            route="test",
            threshold=0.5,
        )

    assert result.status == CacheStatus.HIT
    assert result.cached_answer == "The full plan text"


# ── Delete by id, not by normalized_query (bug fix) ──────────────────────────

def test_expired_exact_entry_deleted_by_id_not_normalized_query(cache_db):
    """
    When find_exact_cached_answer encounters an expired entry it schedules
    _delete_by_id(row_id) — not _delete_by_normalized_query — so a
    concurrently written fresh entry with the same text is not erased.
    """
    import src.services.semantic_cache as mod
    from src.services.semantic_cache import find_exact_cached_answer

    deleted_ids = []

    def capture_delete(row_id):
        deleted_ids.append(row_id)

    # Insert an expired row
    with sqlite3.connect(cache_db) as conn:
        conn.execute(
            "INSERT INTO semantic_cache "
            "(query, normalized_query, answer, route, embedding_json, "
            " source, ttl_days, created_at) "
            "VALUES ('old q', 'old q', 'old ans', 'test', '[]', 'db', 1, "
            "        datetime('now', '-5 days'))",
        )
        row_id = conn.execute(
            "SELECT id FROM semantic_cache WHERE normalized_query = 'old q'"
        ).fetchone()[0]

    with patch("src.services.semantic_cache._bg_submit") as mock_submit:
        find_exact_cached_answer("old q", route="test")

    # Must submit _delete_by_id (not _delete_by_normalized_query)
    assert mock_submit.called
    fn_arg = mock_submit.call_args[0][0]
    assert fn_arg.__name__ == "_delete_by_id"
    id_arg = mock_submit.call_args[0][1]
    assert id_arg == row_id


# ── _extract_travel_start_date ────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("fly to paris on 2026-06-15",               "2026-06-15"),
    ("trip starting 2026-07-04",                  "2026-07-04"),
    ("paris on 20/06/2026",                       "2026-06-20"),
    ("paris on 15/06/2026",                       "2026-06-15"),
    ("paris on 06/15/2026",                       "2026-06-15"),  # US MM/DD fallback
    ("june 15 trip to berlin",                    "2026-06-15"),
    ("15 june trip to berlin",                    "2026-06-15"),
    ("15th of june trip",                         "2026-06-15"),
    ("trip on august 3rd",                        "2026-08-03"),
    ("weekend trip, no date mentioned",           None),
])
def test_extract_travel_start_date(text, expected):
    from src.agents.context_enricher import _extract_travel_start_date
    # Freeze "today" to a date before all test dates so future-date logic is stable
    frozen_today = date(2026, 1, 1)
    with patch("src.agents.context_enricher.date") as mock_date:
        mock_date.today.return_value = frozen_today
        mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
        mock_date.fromisoformat = date.fromisoformat
        result = _extract_travel_start_date(text)
    assert result == expected


# ── _compute_travel_end_date ──────────────────────────────────────────────────

@pytest.mark.parametrize("start,days,expected", [
    ("2026-06-15", 7,  "2026-06-21"),   # 7-day trip: day 1 = 15, day 7 = 21
    ("2026-06-15", 1,  "2026-06-15"),   # 1-day trip
    ("2026-12-30", 5,  "2027-01-03"),   # year boundary
    (None,         7,  None),
    ("2026-06-15", None, None),
])
def test_compute_travel_end_date(start, days, expected):
    from src.agents.context_enricher import _compute_travel_end_date
    assert _compute_travel_end_date(start, days) == expected


# ── Full round-trip: store with dates → lookup respects valid_until ───────────

def test_full_roundtrip_valid_until_expiry(cache_db):
    from src.services.semantic_cache import store_cache_entry, find_exact_cached_answer
    from src.models.cache import CacheStatus

    # Store entry with a travel date in the past
    past_date = (date.today() - timedelta(days=2)).isoformat()
    ctx = {"travel_start_date": past_date, "destination_city": "Paris"}

    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        store_cache_entry("paris trip last week", "old plan", route="test",
                          trip_context=ctx)

    # valid_until = past_date - 1 day (even further in the past) → invisible
    result = find_exact_cached_answer("paris trip last week", route="test")
    assert result.status == CacheStatus.MISS


def test_full_roundtrip_future_date_still_hits(cache_db):
    from src.services.semantic_cache import store_cache_entry, find_exact_cached_answer
    from src.models.cache import CacheStatus

    future_date = (date.today() + timedelta(days=30)).isoformat()
    ctx = {"travel_start_date": future_date, "destination_city": "Tokyo"}

    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        store_cache_entry("tokyo trip next month", "fresh plan", route="test",
                          trip_context=ctx)

    result = find_exact_cached_answer("tokyo trip next month", route="test")
    assert result.status == CacheStatus.HIT
    assert result.cached_answer == "fresh plan"
