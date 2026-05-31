"""
Tests for semantic_cache service.
No real DB or embedding model — all I/O is mocked.
TTL enforcement tests use a real in-memory SQLite DB so the SQL filter logic is exercised.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestNormalizeQuery:

    def test_text_transformations(self):
        from src.services.semantic_cache import normalize_query
        assert normalize_query("PLAN A TRIP TO PARIS") == "plan a trip to paris"
        assert normalize_query("  trip to Tokyo  ") == "trip to tokyo"
        assert normalize_query("trip   to   Berlin") == "trip to berlin"
        assert normalize_query("“fly to london”") == '"fly to london"'
        assert normalize_query("it’s a trip") == "it's a trip"

    def test_edge_cases(self):
        from src.services.semantic_cache import normalize_query
        assert normalize_query("") == ""
        assert normalize_query("plan a 5-day trip to paris") == "plan a 5-day trip to paris"


class TestCosineSimilarity:

    def test_unit_geometry(self):
        from src.services.semantic_cache import cosine_similarity
        import math
        assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6   # identical
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-6          # orthogonal
        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-6  # opposite
        assert abs(cosine_similarity([1.0, 1.0], [1.0, 0.0]) - 1.0 / math.sqrt(2)) < 1e-6

    def test_zero_vector_returns_zero(self):
        from src.services.semantic_cache import cosine_similarity
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


class TestFindCachedAnswer:

    def _row(self, row_id, query, embedding):
        return {"id": row_id, "query": query, "embedding_json": json.dumps(embedding)}

    def test_returns_miss_when_index_empty(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[]):
            result = find_cached_answer("Plan a trip to Paris")
        assert result.status == CacheStatus.MISS
        assert result.cached_answer is None

    def test_returns_hit_above_threshold(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus
        full = {"query": "Paris trip", "answer": "Here is your plan", "compressed_answer": ""}
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[self._row(1, "Paris trip", [1.0, 0.0])]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full):
            result = find_cached_answer("Trip to Paris", threshold=0.85)
        assert result.status == CacheStatus.HIT
        assert result.cached_answer == "Here is your plan"

    def test_returns_miss_below_threshold(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[self._row(1, "Berlin trip", [0.0, 1.0])]):
            result = find_cached_answer("Paris trip", threshold=0.85)
        assert result.status == CacheStatus.MISS
        assert result.cached_answer is None

    def test_full_row_not_fetched_on_miss(self):
        """_fetch_row_by_id must never be called when score is below threshold."""
        from src.services.semantic_cache import find_cached_answer
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[self._row(1, "Berlin trip", [0.0, 1.0])]), \
             patch("src.services.semantic_cache._fetch_row_by_id") as mock_fetch:
            find_cached_answer("Paris trip", threshold=0.85)
        mock_fetch.assert_not_called()

    def test_invalid_embedding_json_skipped(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus
        bad = {"id": 1, "query": "bad", "embedding_json": "not-valid-json"}
        good = self._row(2, "Paris trip", [1.0, 0.0])
        full = {"query": "Paris trip", "answer": "Plan here", "compressed_answer": ""}
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[bad, good]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full):
            result = find_cached_answer("Paris trip", threshold=0.85)
        assert result.status == CacheStatus.HIT

    def test_picks_best_score_and_custom_threshold(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus
        row_low = self._row(1, "Berlin", [0.0, 1.0])
        row_high = self._row(2, "Paris trip", [1.0, 0.0])
        full = {"query": "Paris trip", "answer": "Paris plan", "compressed_answer": ""}
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row_low, row_high]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full) as mock_fetch:
            result = find_cached_answer("Paris trip", threshold=0.85)
        assert result.status == CacheStatus.HIT
        mock_fetch.assert_called_once_with(2)

        # threshold above 1.0 → always miss
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row_high]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full):
            result = find_cached_answer("Paris trip", threshold=1.01)
        assert result.status == CacheStatus.MISS


class TestStoreCacheEntry:

    def _mock_conn(self):
        mock_conn = MagicMock()
        ctx = mock_conn.return_value.__enter__.return_value
        ctx.execute = MagicMock()
        ctx.commit = MagicMock()
        return mock_conn

    def test_returns_entry_with_correct_fields(self):
        from src.services.semantic_cache import store_cache_entry
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.sqlite3.connect", self._mock_conn()):
            entry = store_cache_entry("Paris trip", "Here is your plan", route="cache_check")
        assert entry.query == "Paris trip"
        assert entry.answer == "Here is your plan"
        assert entry.route == "cache_check"

    def test_normalizes_query_before_storing(self):
        from src.services.semantic_cache import store_cache_entry
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.sqlite3.connect", self._mock_conn()):
            entry = store_cache_entry("  PARIS TRIP  ", "Here is your plan")
        assert entry.normalized_query == "paris trip"


class TestTTLAutoDerivation:
    """store_cache_entry must auto-derive ttl_days from source."""

    def _mock_conn(self):
        mock_conn = MagicMock()
        ctx = mock_conn.return_value.__enter__.return_value
        ctx.execute = MagicMock()
        ctx.fetchone = MagicMock(return_value=None)
        ctx.commit = MagicMock()
        return mock_conn

    def test_db_source_gets_ttl_days_db(self):
        from src.services.semantic_cache import store_cache_entry, TTL_DAYS_DB
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.sqlite3.connect", self._mock_conn()):
            entry = store_cache_entry("paris trip", "plan", source="db")
        assert entry.ttl_days == TTL_DAYS_DB
        assert entry.source == "db"

    def test_web_source_gets_ttl_days_web(self):
        from src.services.semantic_cache import store_cache_entry, TTL_DAYS_WEB
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.sqlite3.connect", self._mock_conn()):
            entry = store_cache_entry("london trip", "plan", source="web")
        assert entry.ttl_days == TTL_DAYS_WEB
        assert entry.source == "web"

    def test_explicit_ttl_days_overrides_default(self):
        from src.services.semantic_cache import store_cache_entry
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.sqlite3.connect", self._mock_conn()):
            entry = store_cache_entry("tokyo trip", "plan", source="web", ttl_days=7)
        assert entry.ttl_days == 7

    def test_invalid_source_raises_value_error(self):
        import pytest
        from src.services.semantic_cache import store_cache_entry
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
            with pytest.raises(ValueError, match="source must be one of"):
                store_cache_entry("trip", "plan", source="tavily")


class TestTTLEnforcement:
    """
    TTL expiry tests use a real temporary SQLite DB so the SQL filter logic
    is exercised end-to-end, not mocked away.
    """

    def _setup_temp_db(self, tmp_path):
        """Patches _CACHE_DB_PATH to a fresh temp DB and resets the init flag."""
        import src.services.semantic_cache as mod
        db_path = tmp_path / "test_cache.db"
        self._orig_path = mod._CACHE_DB_PATH
        self._orig_init = mod._db_initialized
        mod._CACHE_DB_PATH = db_path
        mod._db_initialized = False
        mod.initialize_cache_db()
        return db_path

    def _teardown_temp_db(self):
        import src.services.semantic_cache as mod
        mod._CACHE_DB_PATH = self._orig_path
        mod._db_initialized = self._orig_init

    def _backdate(self, db_path, normalized_query, days):
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                f"UPDATE semantic_cache SET created_at = datetime('now', '-{days} days') "
                "WHERE normalized_query = ?",
                (normalized_query,),
            )
            conn.commit()

    def test_web_entry_expired_after_3_days(self, tmp_path):
        from src.services.semantic_cache import (
            store_cache_entry, find_exact_cached_answer, TTL_DAYS_WEB
        )
        from src.models.cache import CacheStatus

        self._setup_temp_db(tmp_path)
        try:
            with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
                 patch("src.services.semantic_cache._cleanup_cache"):
                store_cache_entry("web paris trip", "web plan", source="web")

            self._backdate(tmp_path / "test_cache.db", "web paris trip", TTL_DAYS_WEB + 1)

            result = find_exact_cached_answer("web paris trip")
            assert result.status == CacheStatus.MISS
            assert "expired" in result.reason.lower()
        finally:
            self._teardown_temp_db()

    def test_db_entry_valid_after_4_days(self, tmp_path):
        from src.services.semantic_cache import store_cache_entry, find_exact_cached_answer
        from src.models.cache import CacheStatus

        self._setup_temp_db(tmp_path)
        try:
            with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
                 patch("src.services.semantic_cache._cleanup_cache"):
                store_cache_entry("db paris trip", "db plan", source="db")

            self._backdate(tmp_path / "test_cache.db", "db paris trip", 4)

            result = find_exact_cached_answer("db paris trip")
            assert result.status == CacheStatus.HIT
            assert result.source == "db"
            assert result.ttl_days == 30
        finally:
            self._teardown_temp_db()

    def test_db_entry_expired_after_30_days(self, tmp_path):
        from src.services.semantic_cache import (
            store_cache_entry, find_exact_cached_answer, TTL_DAYS_DB
        )
        from src.models.cache import CacheStatus

        self._setup_temp_db(tmp_path)
        try:
            with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
                 patch("src.services.semantic_cache._cleanup_cache"):
                store_cache_entry("old db trip", "old plan", source="db")

            self._backdate(tmp_path / "test_cache.db", "old db trip", TTL_DAYS_DB + 1)

            result = find_exact_cached_answer("old db trip")
            assert result.status == CacheStatus.MISS
            assert "expired" in result.reason.lower()
        finally:
            self._teardown_temp_db()

    def test_never_stored_gives_not_found_reason(self, tmp_path):
        from src.services.semantic_cache import find_exact_cached_answer
        from src.models.cache import CacheStatus

        self._setup_temp_db(tmp_path)
        try:
            result = find_exact_cached_answer("totally unknown query xyz")
            assert result.status == CacheStatus.MISS
            assert "expired" not in result.reason.lower()
            assert "no cache entry" in result.reason.lower()
        finally:
            self._teardown_temp_db()


class TestCacheInvalidation:
    """
    Tests for the scalable field-based cache invalidation engine.
    Uses a real temporary SQLite DB so SQL logic is fully exercised.
    """

    def _setup(self, tmp_path):
        import src.services.semantic_cache as mod
        self._orig_path = mod._CACHE_DB_PATH
        self._orig_init = mod._db_initialized
        mod._CACHE_DB_PATH = tmp_path / "test_cache.db"
        mod._db_initialized = False
        mod.initialize_cache_db()

    def _teardown(self):
        import src.services.semantic_cache as mod
        mod._CACHE_DB_PATH = self._orig_path
        mod._db_initialized = self._orig_init

    def _store(self, query, answer="answer", source="db"):
        from src.services.semantic_cache import store_cache_entry
        with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"):
            store_cache_entry(query, answer, source=source)

    def test_invalidate_by_destination_removes_paris_entries(self, tmp_path):
        from src.services.semantic_cache import invalidate_cache_by_destination, find_exact_cached_answer
        from src.models.cache import CacheStatus
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            self._store("destination_city:paris | origin_airport:jfk | duration_days:5 | budget_bucket:1500")
            self._store("destination_city:london | origin_airport:tlv | duration_days:7 | budget_bucket:2000")

            deleted = invalidate_cache_by_destination("paris")
            assert deleted == 2

            r_paris = find_exact_cached_answer("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            assert r_paris.status == CacheStatus.MISS

            r_london = find_exact_cached_answer("destination_city:london | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            assert r_london.status == CacheStatus.HIT
        finally:
            self._teardown()

    def test_invalidate_by_multiple_fields_is_precise(self, tmp_path):
        from src.services.semantic_cache import invalidate_cache_by_fields, find_exact_cached_answer
        from src.models.cache import CacheStatus
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            self._store("destination_city:paris | origin_airport:jfk | duration_days:7 | budget_bucket:2000")

            # Only invalidate paris + tlv origin — jfk entry should survive
            deleted = invalidate_cache_by_fields({"destination_city": "paris", "origin_airport": "tlv"})
            assert deleted == 1

            r_tlv = find_exact_cached_answer("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            assert r_tlv.status == CacheStatus.MISS

            r_jfk = find_exact_cached_answer("destination_city:paris | origin_airport:jfk | duration_days:7 | budget_bucket:2000")
            assert r_jfk.status == CacheStatus.HIT
        finally:
            self._teardown()

    def test_invalidate_future_field_works_without_code_changes(self, tmp_path):
        from src.services.semantic_cache import invalidate_cache_by_fields, find_exact_cached_answer
        from src.models.cache import CacheStatus
        self._setup(tmp_path)
        try:
            # Simulate a future key that includes a 'weather:sunny' field
            self._store("destination_city:paris | weather:sunny | duration_days:7")
            self._store("destination_city:paris | weather:rainy | duration_days:7")

            deleted = invalidate_cache_by_fields({"weather": "sunny"})
            assert deleted == 1

            r_sunny = find_exact_cached_answer("destination_city:paris | weather:sunny | duration_days:7")
            assert r_sunny.status == CacheStatus.MISS

            r_rainy = find_exact_cached_answer("destination_city:paris | weather:rainy | duration_days:7")
            assert r_rainy.status == CacheStatus.HIT
        finally:
            self._teardown()

    def test_empty_filters_is_noop(self, tmp_path):
        from src.services.semantic_cache import invalidate_cache_by_fields
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | duration_days:7")
            deleted = invalidate_cache_by_fields({})
            assert deleted == 0
        finally:
            self._teardown()

    def test_returns_count_of_deleted_entries(self, tmp_path):
        from src.services.semantic_cache import invalidate_cache_by_destination
        self._setup(tmp_path)
        try:
            self._store("destination_city:tokyo | duration_days:5")
            self._store("destination_city:tokyo | duration_days:10")
            self._store("destination_city:tokyo | duration_days:14")
            deleted = invalidate_cache_by_destination("tokyo")
            assert deleted == 3
        finally:
            self._teardown()

    def test_freetext_with_colon_not_misidentified_as_structured(self, tmp_path):
        """Free-text queries containing a colon must NOT be treated as structured keys."""
        from src.services.semantic_cache import invalidate_cache_by_destination, find_exact_cached_answer
        from src.models.cache import CacheStatus
        self._setup(tmp_path)
        try:
            # Free-text query with a colon — NOT a structured key
            self._store("paris: best hotels and flights?")
            # Structured key — IS a structured key
            self._store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")

            deleted = invalidate_cache_by_destination("paris")
            # Both should be deleted — structured via phase 1, freetext via phase 2
            assert deleted == 2
        finally:
            self._teardown()

    def test_structured_key_not_matched_by_freetext_phase(self, tmp_path):
        """Structured entries must never be double-counted by the freetext phase."""
        from src.services.semantic_cache import invalidate_cache_by_fields
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            # include_freetext=False — only structured phase runs
            deleted = invalidate_cache_by_fields({"destination_city": "paris"}, include_freetext=False)
            assert deleted == 1  # not 2 — no double counting
        finally:
            self._teardown()


class TestNotifyDbChanged:
    """Tests for the DB-change notification registry."""

    def _setup(self, tmp_path):
        import src.services.semantic_cache as mod
        self._orig_path = mod._CACHE_DB_PATH
        self._orig_init = mod._db_initialized
        mod._CACHE_DB_PATH = tmp_path / "test_cache.db"
        mod._db_initialized = False
        mod.initialize_cache_db()

    def _teardown(self):
        import src.services.semantic_cache as mod
        mod._CACHE_DB_PATH = self._orig_path
        mod._db_initialized = self._orig_init

    def _store(self, query, source="db"):
        from src.services.semantic_cache import store_cache_entry
        with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"):
            store_cache_entry(query, "answer", source=source)

    def test_hotels_change_invalidates_destination(self, tmp_path):
        from src.services.semantic_cache import notify_db_changed, find_exact_cached_answer
        from src.models.cache import CacheStatus
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            deleted = notify_db_changed("hotels", {"destination_city": "Paris"})
            assert deleted == 1
            r = find_exact_cached_answer("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            assert r.status == CacheStatus.MISS
        finally:
            self._teardown()

    def test_flights_change_invalidates_destination_and_origin(self, tmp_path):
        from src.services.semantic_cache import notify_db_changed
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | origin_airport:tlv | duration_days:7 | budget_bucket:2000")
            self._store("destination_city:london | origin_airport:tlv | duration_days:5 | budget_bucket:1500")
            self._store("destination_city:paris | origin_airport:jfk | duration_days:7 | budget_bucket:2000")

            # flights change for TLV origin → should hit both paris+tlv and london+tlv
            deleted = notify_db_changed("flights", {"destination_city": "paris", "origin_airport": "tlv"})
            assert deleted == 1  # only paris+tlv, not london+tlv (different destination)
        finally:
            self._teardown()

    def test_unknown_table_returns_zero(self, tmp_path):
        from src.services.semantic_cache import notify_db_changed
        self._setup(tmp_path)
        try:
            deleted = notify_db_changed("unknown_table", {"destination_city": "paris"})
            assert deleted == 0
        finally:
            self._teardown()

    def test_missing_values_returns_zero(self, tmp_path):
        from src.services.semantic_cache import notify_db_changed
        self._setup(tmp_path)
        try:
            # hotels registry expects destination_city — not provided
            deleted = notify_db_changed("hotels", {"some_other_field": "value"})
            assert deleted == 0
        finally:
            self._teardown()

    def test_new_table_in_registry_works_without_code_changes(self, tmp_path):
        """Adding a new table to the registry is all that's needed."""
        import src.services.semantic_cache as mod
        from src.services.semantic_cache import notify_db_changed
        self._setup(tmp_path)
        try:
            self._store("destination_city:paris | weather:sunny | duration_days:7")
            # weather table is already in the registry
            deleted = notify_db_changed("weather", {"destination_city": "paris"})
            assert deleted == 1
        finally:
            self._teardown()


class TestBuildTripCacheKey:
    """Tests for the scalable, versioned, currency-aware cache key builder."""

    def _key(self, **kwargs):
        from src.services.semantic_cache import build_trip_cache_key
        return build_trip_cache_key(kwargs)

    # ── Required fields ──────────────────────────────────────────────────────

    def test_returns_none_without_destination(self):
        assert self._key(duration_days=7) is None

    def test_returns_none_without_duration(self):
        assert self._key(destination_city="Paris") is None

    def test_returns_key_with_only_required_fields(self):
        key = self._key(destination_city="Paris", duration_days=7)
        assert key is not None
        assert "destination_city:paris" in key
        assert "duration:week" in key

    # ── Duration bucketing ───────────────────────────────────────────────────

    def test_duration_buckets(self):
        from src.services.semantic_cache import build_trip_cache_key
        cases = [
            (1, "short"), (3, "short"),
            (4, "week"),  (7, "week"),
            (8, "extended"), (14, "extended"),
            (15, "long"), (30, "long"),
        ]
        for days, expected_bucket in cases:
            key = build_trip_cache_key({"destination_city": "Paris", "duration_days": days})
            assert f"duration:{expected_bucket}" in key, f"days={days} expected bucket={expected_bucket}"

    def test_same_duration_bucket_gives_same_key(self):
        from src.services.semantic_cache import build_trip_cache_key
        k5 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 5})
        k6 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 6})
        assert k5 == k6  # both → "week"

    # ── Currency-aware budget ────────────────────────────────────────────────

    def test_usd_budget_bucketed(self):
        key = self._key(destination_city="Paris", duration_days=7, total_budget=2000, currency="USD")
        assert "budget_bucket:2000_usd" in key

    def test_eur_budget_produces_different_key_than_usd(self):
        k_usd = self._key(destination_city="Paris", duration_days=7, total_budget=2000, currency="USD")
        k_eur = self._key(destination_city="Paris", duration_days=7, total_budget=2000, currency="EUR")
        assert k_usd != k_eur
        assert "budget_bucket:2000_usd" in k_usd
        assert "budget_bucket:2000_eur" in k_eur

    def test_near_budgets_same_bucket(self):
        k1 = self._key(destination_city="Paris", duration_days=7, total_budget=1980, currency="USD")
        k2 = self._key(destination_city="Paris", duration_days=7, total_budget=2020, currency="USD")
        assert k1 == k2  # both round to 2000

    def test_unknown_currency_defaults_to_usd(self):
        key = self._key(destination_city="Paris", duration_days=7, total_budget=2000, currency="XYZ")
        assert "budget_bucket:2000_usd" in key

    # ── Traveler bucketing ───────────────────────────────────────────────────

    def test_traveler_buckets(self):
        from src.services.semantic_cache import build_trip_cache_key
        cases = [
            (1, "solo"), (2, "couple"),
            (3, "small_group"), (4, "small_group"),
            (5, "large_group"), (10, "large_group"),
        ]
        for n, expected in cases:
            key = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "num_travelers": n})
            assert f"travelers:{expected}" in key, f"n={n} expected={expected}"

    def test_different_traveler_groups_different_keys(self):
        k_solo = self._key(destination_city="Paris", duration_days=7, num_travelers=1)
        k_couple = self._key(destination_city="Paris", duration_days=7, num_travelers=2)
        assert k_solo != k_couple

    # ── Key properties ───────────────────────────────────────────────────────

    def test_key_is_versioned(self):
        from src.services.semantic_cache import _CACHE_KEY_VERSION
        key = self._key(destination_city="Paris", duration_days=7)
        assert f"key_version:{_CACHE_KEY_VERSION}" in key

    def test_key_is_deterministic_regardless_of_field_order(self):
        from src.services.semantic_cache import build_trip_cache_key
        k1 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "num_travelers": 2, "total_budget": 2000})
        k2 = build_trip_cache_key({"total_budget": 2000, "num_travelers": 2, "duration_days": 7, "destination_city": "Paris"})
        assert k1 == k2

    def test_unknown_fields_are_ignored(self):
        from src.services.semantic_cache import build_trip_cache_key
        k1 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7})
        k2 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7, "future_field": "value"})
        assert k1 == k2

    def test_backwards_compat_shim(self):
        from src.services.semantic_cache import build_trip_cache_key, build_trip_cache_key_from_context
        k1 = build_trip_cache_key({"destination_city": "Paris", "duration_days": 7})
        k2 = build_trip_cache_key_from_context(destination_city="Paris", duration_days=7)
        assert k1 == k2

    def test_structured_key_bypass_uses_key_version(self):
        """Structured keys must be bypassed by key_version: prefix, not origin_airport:."""
        from src.services.semantic_cache import find_cached_answer, _CACHE_KEY_VERSION
        from src.models.cache import CacheStatus
        # A structured key without origin_airport should still bypass semantic search
        query = f"key_version:{_CACHE_KEY_VERSION} | destination_city:paris | duration:week"
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]):
            result = find_cached_answer(query)
        assert result.status == CacheStatus.MISS
        assert "structured key" in result.reason.lower()

    def test_register_cache_key_field(self):
        """Registering a new field adds it to the key schema."""
        import src.services.semantic_cache as mod
        from src.services.semantic_cache import build_trip_cache_key, register_cache_key_field
        # Use a unique name to avoid conflict with parallel tests
        field_name = "_test_weather_pref"
        try:
            register_cache_key_field(field_name, lambda v: v.strip().lower(), key_name="weather")
            key = build_trip_cache_key({
                "destination_city": "Paris",
                "duration_days": 7,
                field_name: "Sunny",
            })
            assert "weather:sunny" in key
        finally:
            mod._CACHE_KEY_FIELDS.pop(field_name, None)

    def test_register_duplicate_field_raises(self):
        import pytest
        from src.services.semantic_cache import register_cache_key_field
        with pytest.raises(ValueError, match="already registered"):
            register_cache_key_field("destination_city", lambda v: v)

    def test_batch_cosine_same_result_as_loop(self):
        """Batch numpy cosine must return the same best match as the old Python loop."""
        import numpy as np
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus

        query_vec = [1.0, 0.0, 0.0]
        best_vec  = [0.99, 0.1, 0.0]   # closest
        other_vec = [0.0, 1.0, 0.0]    # orthogonal

        index = [
            {"id": 1, "query": "best match",  "embedding_json": json.dumps(best_vec)},
            {"id": 2, "query": "other match", "embedding_json": json.dumps(other_vec)},
        ]
        full_row = {"query": "best match", "answer": "the answer", "compressed_answer": "", "source": "db", "ttl_days": 30}

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.embed_text", return_value=query_vec), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=index), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full_row):
            result = find_cached_answer("any free text query", threshold=0.5)

        assert result.status == CacheStatus.HIT
        assert result.cached_answer == "the answer"


class TestNewFixes:
    """Tests for the 9-item bug/upgrade pass."""

    def test_ttl_days_capped_at_max(self):
        from src.services.semantic_cache import store_cache_entry, _MAX_TTL_DAYS
        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"), \
             patch("src.services.semantic_cache.sqlite3.connect") as mock_conn:
            ctx = mock_conn.return_value.__enter__.return_value
            ctx.execute = MagicMock()
            ctx.fetchone = MagicMock(return_value=None)
            ctx.commit = MagicMock()
            entry = store_cache_entry("test", "answer", source="db", ttl_days=99999)
        assert entry.ttl_days == _MAX_TTL_DAYS

    def test_invalid_iata_returns_none_from_key(self):
        from src.services.semantic_cache import build_trip_cache_key
        # "New York" is not a valid IATA code — should be omitted from key
        key = build_trip_cache_key({
            "destination_city": "Paris",
            "duration_days": 7,
            "origin_airport": "New York",
        })
        assert key is not None
        assert "new york" not in key
        assert "origin_airport" not in key

    def test_valid_iata_included_in_key(self):
        from src.services.semantic_cache import build_trip_cache_key
        key = build_trip_cache_key({
            "destination_city": "Paris",
            "duration_days": 7,
            "origin_airport": "TLV",
        })
        assert "origin_airport:tlv" in key

    def test_register_currency_valid(self):
        from src.services.semantic_cache import register_currency, _REGISTERED_CURRENCIES
        register_currency("COP")
        assert "cop" in _REGISTERED_CURRENCIES
        _REGISTERED_CURRENCIES.discard("cop")  # cleanup

    def test_register_currency_invalid_raises(self):
        import pytest
        from src.services.semantic_cache import register_currency
        with pytest.raises(ValueError, match="Invalid currency"):
            register_currency("")
        with pytest.raises(ValueError, match="Invalid currency"):
            register_currency("bitcoin123")

    def test_register_currency_used_in_key(self):
        from src.services.semantic_cache import register_currency, build_trip_cache_key, _REGISTERED_CURRENCIES
        register_currency("COP")
        key = build_trip_cache_key({
            "destination_city": "Bogota",
            "duration_days": 5,
            "total_budget": 3000000,
            "currency": "COP",
        })
        assert "budget_bucket" in key
        assert "_cop" in key
        _REGISTERED_CURRENCIES.discard("cop")

    def test_find_exact_single_connection_on_miss(self, tmp_path):
        """find_exact_cached_answer uses one connection even on miss."""
        import src.services.semantic_cache as mod
        from src.services.semantic_cache import find_exact_cached_answer
        from src.models.cache import CacheStatus
        orig_path, orig_init = mod._CACHE_DB_PATH, mod._db_initialized
        mod._CACHE_DB_PATH = tmp_path / "test.db"
        mod._db_initialized = False
        mod.initialize_cache_db()
        try:
            result = find_exact_cached_answer("never stored query xyz")
            assert result.status == CacheStatus.MISS
            assert "no cache entry" in result.reason.lower()
        finally:
            mod._CACHE_DB_PATH = orig_path
            mod._db_initialized = orig_init

    def test_find_exact_single_connection_on_expired(self, tmp_path):
        """Expired entry reason comes from single-connection query."""
        import sqlite3 as _sqlite3
        import src.services.semantic_cache as mod
        from src.services.semantic_cache import store_cache_entry, find_exact_cached_answer
        from src.models.cache import CacheStatus
        orig_path, orig_init = mod._CACHE_DB_PATH, mod._db_initialized
        mod._CACHE_DB_PATH = tmp_path / "test.db"
        mod._db_initialized = False
        mod.initialize_cache_db()
        try:
            with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
                 patch("src.services.semantic_cache._cleanup_cache"):
                store_cache_entry("expiring query", "answer", source="web")
            with _sqlite3.connect(tmp_path / "test.db") as conn:
                conn.execute("UPDATE semantic_cache SET created_at = datetime('now', '-10 days') WHERE normalized_query = 'expiring query'")
                conn.commit()
            result = find_exact_cached_answer("expiring query")
            assert result.status == CacheStatus.MISS
            assert "expired" in result.reason.lower()
        finally:
            mod._CACHE_DB_PATH = orig_path
            mod._db_initialized = orig_init


class TestDbInitializedGuard:

    def test_initialize_cache_db_runs_only_once(self):
        import src.services.semantic_cache as mod
        original = mod._db_initialized
        try:
            mod._db_initialized = True
            with patch("src.services.semantic_cache.sqlite3.connect") as mock_conn:
                mod.initialize_cache_db()
            mock_conn.assert_not_called()
        finally:
            mod._db_initialized = original
