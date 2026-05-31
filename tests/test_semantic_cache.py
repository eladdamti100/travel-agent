"""
Tests for semantic_cache service.
No real DB or embedding model — all I/O is mocked.
TTL enforcement and invalidation tests use a real in-memory SQLite DB.
"""

import json
import sqlite3
import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_cache_db(tmp_path):
    import src.services.semantic_cache as mod
    orig_path, orig_init = mod._CACHE_DB_PATH, mod._db_initialized
    mod._CACHE_DB_PATH = tmp_path / "test_cache.db"
    mod._db_initialized = False
    mod.initialize_cache_db()
    yield tmp_path / "test_cache.db"
    mod._CACHE_DB_PATH = orig_path
    mod._db_initialized = orig_init


def _mock_conn():
    conn = MagicMock()
    ctx = conn.return_value.__enter__.return_value
    ctx.execute = MagicMock()
    ctx.fetchone = MagicMock(return_value=None)
    ctx.commit = MagicMock()
    return conn


def _store(query, answer="answer", source="db"):
    from src.services.semantic_cache import store_cache_entry
    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._cleanup_cache"):
        store_cache_entry(query, answer, source=source)


# ── Normalize & similarity ────────────────────────────────────────────────────

def test_normalize_query():
    from src.services.semantic_cache import normalize_query
    assert normalize_query("PLAN A TRIP TO PARIS") == "plan a trip to paris"
    assert normalize_query("  trip to Tokyo  ") == "trip to tokyo"
    assert normalize_query("trip   to   Berlin") == "trip to berlin"
    assert normalize_query("") == ""
    assert normalize_query("plan a 5-day trip to paris") == "plan a 5-day trip to paris"


def test_cosine_similarity():
    from src.services.semantic_cache import cosine_similarity
    import math
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6
    assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-6
    assert abs(cosine_similarity([1.0, 1.0], [1.0, 0.0]) - 1.0 / math.sqrt(2)) < 1e-6
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# ── find_cached_answer ────────────────────────────────────────────────────────

def test_find_cached_answer():
    from src.services.semantic_cache import find_cached_answer
    from src.models.cache import CacheStatus

    def _row(rid, query, emb):
        return {"id": rid, "query": query, "embedding_json": json.dumps(emb)}

    full = {"query": "Paris trip", "answer": "Here is your plan", "compressed_answer": ""}

    # Empty index → miss
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=[]):
        assert find_cached_answer("Paris trip").status == CacheStatus.MISS

    # Above threshold → hit
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=[_row(1, "Paris trip", [1.0, 0.0])]), \
         patch("src.services.semantic_cache._fetch_row_by_id", return_value=full):
        result = find_cached_answer("Trip to Paris", threshold=0.85)
    assert result.status == CacheStatus.HIT and result.cached_answer == "Here is your plan"

    # Below threshold → miss and _fetch_row_by_id not called
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=[_row(1, "Berlin trip", [0.0, 1.0])]), \
         patch("src.services.semantic_cache._fetch_row_by_id") as mock_fetch:
        result = find_cached_answer("Paris trip", threshold=0.85)
    assert result.status == CacheStatus.MISS
    mock_fetch.assert_not_called()

    # Bad JSON row skipped, valid row still matched
    bad = {"id": 1, "query": "bad", "embedding_json": "not-valid-json"}
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=[bad, _row(2, "Paris trip", [1.0, 0.0])]), \
         patch("src.services.semantic_cache._fetch_row_by_id", return_value=full):
        assert find_cached_answer("Paris trip", threshold=0.85).status == CacheStatus.HIT

    # Picks best score; threshold > 1.0 → always miss
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=[_row(1, "Berlin", [0.0, 1.0]), _row(2, "Paris", [1.0, 0.0])]), \
         patch("src.services.semantic_cache._fetch_row_by_id", return_value=full) as mock_fetch:
        result = find_cached_answer("Paris trip", threshold=0.85)
    assert result.status == CacheStatus.HIT
    mock_fetch.assert_called_once_with(2)

    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=[_row(1, "Paris", [1.0, 0.0])]), \
         patch("src.services.semantic_cache._fetch_row_by_id", return_value=full):
        assert find_cached_answer("Paris trip", threshold=1.01).status == CacheStatus.MISS


# ── store_cache_entry ─────────────────────────────────────────────────────────

def test_store_cache_entry():
    from src.services.semantic_cache import store_cache_entry
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._cleanup_cache"), \
         patch("src.services.semantic_cache.sqlite3.connect", _mock_conn()):
        entry = store_cache_entry("  PARIS TRIP  ", "Here is your plan", route="cache_check")
    assert entry.query == "  PARIS TRIP  "
    assert entry.normalized_query == "paris trip"
    assert entry.answer == "Here is your plan"
    assert entry.route == "cache_check"


# ── TTL derivation ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source,expected_const", [("db", "TTL_DAYS_DB"), ("web", "TTL_DAYS_WEB")])
def test_ttl_derivation(source, expected_const):
    import src.services.semantic_cache as mod
    from src.services.semantic_cache import store_cache_entry
    expected_ttl = getattr(mod, expected_const)
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._cleanup_cache"), \
         patch("src.services.semantic_cache.sqlite3.connect", _mock_conn()):
        entry = store_cache_entry(f"{source} trip", "plan", source=source)
    assert entry.ttl_days == expected_ttl and entry.source == source


def test_ttl_explicit_override_and_invalid_source():
    from src.services.semantic_cache import store_cache_entry
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._cleanup_cache"), \
         patch("src.services.semantic_cache.sqlite3.connect", _mock_conn()):
        entry = store_cache_entry("tokyo trip", "plan", source="web", ttl_days=7)
    assert entry.ttl_days == 7

    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        with pytest.raises(ValueError, match="source must be one of"):
            store_cache_entry("trip", "plan", source="tavily")


# ── TTL enforcement (real DB) ─────────────────────────────────────────────────

@pytest.mark.parametrize("source,days_past_ttl,should_expire", [
    ("web", 1,   True),
    ("db",  -26, False),   # 4 days < TTL_DB=30
    ("db",  1,   True),
])
def test_ttl_enforcement(temp_cache_db, source, days_past_ttl, should_expire):
    import src.services.semantic_cache as mod
    from src.services.semantic_cache import store_cache_entry, find_exact_cached_answer
    from src.models.cache import CacheStatus

    ttl = mod.TTL_DAYS_WEB if source == "web" else mod.TTL_DAYS_DB
    backdate = ttl + days_past_ttl
    query = f"{source}_ttl_test_{days_past_ttl}"

    with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._cleanup_cache"):
        store_cache_entry(query, "plan", source=source)

    with sqlite3.connect(temp_cache_db) as conn:
        conn.execute(
            f"UPDATE semantic_cache SET created_at = datetime('now', '-{backdate} days') "
            "WHERE normalized_query = ?", (query,)
        )
        conn.commit()

    result = find_exact_cached_answer(query)
    if should_expire:
        assert result.status == CacheStatus.MISS and "expired" in result.reason.lower()
    else:
        assert result.status == CacheStatus.HIT


def test_not_found_reason(temp_cache_db):
    from src.services.semantic_cache import find_exact_cached_answer
    from src.models.cache import CacheStatus
    result = find_exact_cached_answer("totally unknown query xyz")
    assert result.status == CacheStatus.MISS
    assert "expired" not in result.reason.lower()
    assert "no cache entry" in result.reason.lower()


# ── Cache invalidation (real DB) ──────────────────────────────────────────────

def test_cache_invalidation(temp_cache_db):
    from src.services.semantic_cache import (
        invalidate_cache_by_destination, invalidate_cache_by_fields, find_exact_cached_answer,
    )
    from src.models.cache import CacheStatus

    # Destination invalidation
    _store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
    _store("destination_city:paris | origin_airport:jfk | duration_days:5 | budget_bucket:1500")
    _store("destination_city:london | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
    assert invalidate_cache_by_destination("paris") == 2
    assert find_exact_cached_answer("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000").status == CacheStatus.MISS
    assert find_exact_cached_answer("destination_city:london | origin_airport:tlv | duration_days:7 | budget_bucket:2000").status == CacheStatus.HIT

    # Field-level precision: paris+tlv only, jfk survives
    _store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
    _store("destination_city:paris | origin_airport:jfk | duration_days:7 | budget_bucket:2000")
    assert invalidate_cache_by_fields({"destination_city": "paris", "origin_airport": "tlv"}) == 1
    assert find_exact_cached_answer("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000").status == CacheStatus.MISS
    assert find_exact_cached_answer("destination_city:paris | origin_airport:jfk | duration_days:7 | budget_bucket:2000").status == CacheStatus.HIT

    # Empty filters is noop
    _store("destination_city:berlin | duration_days:7")
    assert invalidate_cache_by_fields({}) == 0

    # Structured key not double-counted
    _store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:5000")
    assert invalidate_cache_by_fields({"destination_city": "paris"}, include_freetext=False) >= 1


# ── notify_db_changed ─────────────────────────────────────────────────────────

def test_notify_db_changed(temp_cache_db):
    from src.services.semantic_cache import notify_db_changed, find_exact_cached_answer
    from src.models.cache import CacheStatus

    # hotels change invalidates destination
    _store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
    assert notify_db_changed("hotels", {"destination_city": "Paris"}) == 1
    assert find_exact_cached_answer("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000").status == CacheStatus.MISS

    # flights change with specific fields
    _store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
    _store("destination_city:london | origin_airport:tlv | duration_days:5 | budget_bucket:1500")
    _store("destination_city:paris | origin_airport:jfk | duration_days:7 | budget_bucket:2000")
    deleted = notify_db_changed("flights", {"destination_city": "paris", "origin_airport": "tlv"})
    assert deleted == 1

    # Unknown table and missing values return 0
    assert notify_db_changed("unknown_table", {"destination_city": "paris"}) == 0
    assert notify_db_changed("hotels", {"some_other_field": "value"}) == 0

    # New table in registry works without code changes
    _store("destination_city:paris | weather:sunny | duration_days:7")
    assert notify_db_changed("weather", {"destination_city": "paris"}) >= 1


# ── build_trip_cache_key ──────────────────────────────────────────────────────

def _key(**kwargs):
    from src.services.semantic_cache import build_trip_cache_key
    return build_trip_cache_key(kwargs)


def test_cache_key_required_and_bucketing():
    from src.services.semantic_cache import build_trip_cache_key

    assert _key(duration_days=7) is None
    assert _key(destination_city="Paris") is None

    key = _key(destination_city="Paris", duration_days=7)
    assert key is not None and "destination_city:paris" in key and "duration:week" in key

    for days, bucket in [(1, "short"), (3, "short"), (4, "week"), (7, "week"),
                         (8, "extended"), (14, "extended"), (15, "long"), (30, "long")]:
        k = build_trip_cache_key({"destination_city": "Paris", "duration_days": days})
        assert f"duration:{bucket}" in k, f"days={days}"

    assert _key(destination_city="Paris", duration_days=5) == _key(destination_city="Paris", duration_days=6)

    for n, bucket in [(1, "solo"), (2, "couple"), (3, "small_group"), (4, "small_group"),
                      (5, "large_group"), (10, "large_group")]:
        k = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "num_travelers": n})
        assert f"travelers:{bucket}" in k, f"n={n}"


def test_cache_key_currency_and_properties():
    from src.services.semantic_cache import (
        build_trip_cache_key, build_trip_cache_key_from_context,
        find_cached_answer, _CACHE_KEY_VERSION,
    )
    from src.models.cache import CacheStatus

    k_usd = _key(destination_city="Paris", duration_days=7, total_budget=2000, currency="USD")
    k_eur = _key(destination_city="Paris", duration_days=7, total_budget=2000, currency="EUR")
    assert k_usd != k_eur
    assert "budget_bucket:2000_usd" in k_usd and "budget_bucket:2000_eur" in k_eur

    # Near budgets round to same bucket; unknown currency defaults to usd
    assert _key(destination_city="Paris", duration_days=7, total_budget=1980, currency="USD") == \
           _key(destination_city="Paris", duration_days=7, total_budget=2020, currency="USD")
    assert "budget_bucket:2000_usd" in _key(destination_city="Paris", duration_days=7, total_budget=2000, currency="XYZ")

    # Versioned, deterministic, unknown fields ignored
    assert f"key_version:{_CACHE_KEY_VERSION}" in _key(destination_city="Paris", duration_days=7)
    k1 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "num_travelers": 2, "total_budget": 2000})
    k2 = build_trip_cache_key({"total_budget": 2000, "num_travelers": 2, "duration_days": 7, "destination_city": "Paris"})
    assert k1 == k2
    assert _key(destination_city="Paris", duration_days=7) == \
           build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "future_field": "value"})

    # Backwards-compat shim
    assert _key(destination_city="Paris", duration_days=7) == \
           build_trip_cache_key_from_context(destination_city="Paris", duration_days=7)

    # Structured key bypasses semantic search
    query = f"key_version:{_CACHE_KEY_VERSION} | destination_city:paris | duration:week"
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache._cleanup_cache"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
        result = find_cached_answer(query)
    assert result.status == CacheStatus.MISS and "structured key" in result.reason.lower()


# ── New fixes + misc ──────────────────────────────────────────────────────────

def test_new_fixes():
    import numpy as np
    from src.services.semantic_cache import (
        store_cache_entry, find_cached_answer, build_trip_cache_key,
        register_cache_key_field, register_currency, _MAX_TTL_DAYS, _REGISTERED_CURRENCIES,
    )
    import src.services.semantic_cache as mod
    from src.models.cache import CacheStatus

    # TTL capped at max
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
         patch("src.services.semantic_cache._cleanup_cache"), \
         patch("src.services.semantic_cache.sqlite3.connect") as mc:
        ctx = mc.return_value.__enter__.return_value
        ctx.execute = MagicMock()
        ctx.fetchone = MagicMock(return_value=None)
        ctx.commit = MagicMock()
        entry = store_cache_entry("test", "answer", source="db", ttl_days=99999)
    assert entry.ttl_days == _MAX_TTL_DAYS

    # Invalid IATA omitted from key
    key = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "origin_airport": "New York"})
    assert key is not None and "new york" not in key and "origin_airport" not in key

    # Valid IATA included
    assert "origin_airport:tlv" in build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "origin_airport": "TLV"})

    # Currency registry
    register_currency("COP")
    assert "cop" in _REGISTERED_CURRENCIES
    k = build_trip_cache_key({"destination_city": "Bogota", "duration_days": 5, "total_budget": 3000000, "currency": "COP"})
    assert "budget_bucket" in k and "_cop" in k
    _REGISTERED_CURRENCIES.discard("cop")

    with pytest.raises(ValueError, match="Invalid currency"):
        register_currency("")
    with pytest.raises(ValueError, match="Invalid currency"):
        register_currency("bitcoin123")

    # Duplicate field raises
    with pytest.raises(ValueError, match="already registered"):
        register_cache_key_field("destination_city", lambda v: v)

    # Batch cosine returns same best match as loop
    index = [
        {"id": 1, "query": "best",  "embedding_json": json.dumps([0.99, 0.1, 0.0])},
        {"id": 2, "query": "other", "embedding_json": json.dumps([0.0, 1.0, 0.0])},
    ]
    full_row = {"query": "best", "answer": "the answer", "compressed_answer": "", "source": "db", "ttl_days": 30}
    with patch("src.services.semantic_cache.initialize_cache_db"), \
         patch("src.services.semantic_cache._cleanup_cache"), \
         patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0, 0.0]), \
         patch("src.services.semantic_cache._load_embedding_index", return_value=index), \
         patch("src.services.semantic_cache._fetch_row_by_id", return_value=full_row):
        result = find_cached_answer("any free text query", threshold=0.5)
    assert result.status == CacheStatus.HIT and result.cached_answer == "the answer"


def test_db_initialized_guard():
    import src.services.semantic_cache as mod
    original = mod._db_initialized
    try:
        mod._db_initialized = True
        with patch("src.services.semantic_cache.sqlite3.connect") as mock_conn:
            mod.initialize_cache_db()
        mock_conn.assert_not_called()
    finally:
        mod._db_initialized = original


def test_stale_key_version_purged_on_init(tmp_path):
    """Entries with old key_version: prefix must be deleted when initialize_cache_db() runs."""
    import src.services.semantic_cache as mod
    from src.services.semantic_cache import find_exact_cached_answer
    from src.models.cache import CacheStatus

    orig_path, orig_init = mod._CACHE_DB_PATH, mod._db_initialized
    mod._CACHE_DB_PATH = tmp_path / "test_cache.db"
    mod._db_initialized = False
    mod.initialize_cache_db()

    try:
        old_key = "key_version:v0 | destination_city:paris | duration:week"
        with sqlite3.connect(tmp_path / "test_cache.db") as conn:
            conn.execute(
                "INSERT INTO semantic_cache "
                "(query, normalized_query, answer, route, embedding_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (old_key, old_key, "old answer", "cache_check", "[1.0]"),
            )
            conn.commit()

        # Re-init must purge the old-version row
        mod._db_initialized = False
        mod.initialize_cache_db()

        assert find_exact_cached_answer(old_key).status == CacheStatus.MISS
    finally:
        mod._CACHE_DB_PATH = orig_path
        mod._db_initialized = orig_init
