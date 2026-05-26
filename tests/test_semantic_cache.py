"""
Tests for semantic_cache service.
No real DB or embedding model — all I/O is mocked.
"""

import json
from unittest.mock import patch, MagicMock


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
