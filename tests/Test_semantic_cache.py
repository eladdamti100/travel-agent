"""
Tests for semantic_cache service.
No real DB or embedding model — all I/O is mocked.
"""

import json
from unittest.mock import patch, MagicMock


class TestNormalizeQuery:

    def test_lowercase(self):
        from src.services.semantic_cache import normalize_query
        assert normalize_query("PLAN A TRIP TO PARIS") == "plan a trip to paris"

    def test_strips_leading_trailing_whitespace(self):
        from src.services.semantic_cache import normalize_query
        assert normalize_query("  trip to Tokyo  ") == "trip to tokyo"

    def test_collapses_internal_whitespace(self):
        from src.services.semantic_cache import normalize_query
        assert normalize_query("trip   to   Berlin") == "trip to berlin"

    def test_curly_double_quotes_normalized(self):
        from src.services.semantic_cache import normalize_query
        result = normalize_query("“fly to london”")
        assert result == '"fly to london"'

    def test_curly_single_quotes_normalized(self):
        from src.services.semantic_cache import normalize_query
        result = normalize_query("it’s a trip")
        assert result == "it's a trip"

    def test_empty_string(self):
        from src.services.semantic_cache import normalize_query
        assert normalize_query("") == ""

    def test_already_normalized_unchanged(self):
        from src.services.semantic_cache import normalize_query
        s = "plan a 5-day trip to paris"
        assert normalize_query(s) == s


class TestCosineSimilarity:

    def test_identical_vectors_returns_one(self):
        from src.services.semantic_cache import cosine_similarity
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_returns_zero(self):
        from src.services.semantic_cache import cosine_similarity
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors_returns_minus_one(self):
        from src.services.semantic_cache import cosine_similarity
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        from src.services.semantic_cache import cosine_similarity
        zero = [0.0, 0.0, 0.0]
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(zero, v) == 0.0

    def test_both_zero_vectors_returns_zero(self):
        from src.services.semantic_cache import cosine_similarity
        zero = [0.0, 0.0]
        assert cosine_similarity(zero, zero) == 0.0

    def test_partial_similarity(self):
        from src.services.semantic_cache import cosine_similarity
        import math
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert abs(cosine_similarity(a, b) - expected) < 1e-6


class TestFindCachedAnswer:

    def _make_index_row(self, row_id: int, query: str, embedding: list) -> dict:
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

        row = self._make_index_row(1, "Paris trip", [1.0, 0.0])
        full_row = {"query": "Paris trip", "answer": "Here is your plan", "compressed_answer": "bullet summary"}

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full_row):
            result = find_cached_answer("Trip to Paris", threshold=0.85)

        assert result.status == CacheStatus.HIT
        assert result.cached_answer == "Here is your plan"
        assert abs(result.similarity_score - 1.0) < 1e-6

    def test_returns_miss_below_threshold(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus

        row = self._make_index_row(1, "Berlin trip", [0.0, 1.0])

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row]):
            result = find_cached_answer("Paris trip", threshold=0.85)

        assert result.status == CacheStatus.MISS
        assert result.cached_answer is None

    def test_full_row_not_fetched_on_miss(self):
        """_fetch_row_by_id must never be called when the best score is below threshold."""
        from src.services.semantic_cache import find_cached_answer

        row = self._make_index_row(1, "Berlin trip", [0.0, 1.0])

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row]), \
             patch("src.services.semantic_cache._fetch_row_by_id") as mock_fetch:
            find_cached_answer("Paris trip", threshold=0.85)
            mock_fetch.assert_not_called()

    def test_invalid_embedding_json_skipped(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus

        bad_row = {"id": 1, "query": "bad row", "embedding_json": "not-valid-json"}
        good_row = self._make_index_row(2, "Paris trip", [1.0, 0.0])
        full_row = {"query": "Paris trip", "answer": "Plan here", "compressed_answer": ""}

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[bad_row, good_row]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full_row):
            result = find_cached_answer("Paris trip", threshold=0.85)

        assert result.status == CacheStatus.HIT

    def test_picks_best_score_among_multiple_rows(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus

        row_low = self._make_index_row(1, "Berlin", [0.0, 1.0])
        row_high = self._make_index_row(2, "Paris trip", [1.0, 0.0])
        full_row = {"query": "Paris trip", "answer": "Paris plan", "compressed_answer": ""}

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row_low, row_high]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full_row) as mock_fetch:
            result = find_cached_answer("Paris trip", threshold=0.85)

        assert result.status == CacheStatus.HIT
        mock_fetch.assert_called_once_with(2)

    def test_custom_threshold_respected(self):
        from src.services.semantic_cache import find_cached_answer
        from src.models.cache import CacheStatus

        row = self._make_index_row(1, "Paris trip", [1.0, 0.0])
        full_row = {"query": "Paris trip", "answer": "Paris plan", "compressed_answer": ""}

        with patch("src.services.semantic_cache.initialize_cache_db"), \
             patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0]), \
             patch("src.services.semantic_cache._load_embedding_index", return_value=[row]), \
             patch("src.services.semantic_cache._fetch_row_by_id", return_value=full_row):
            result = find_cached_answer("Paris trip", threshold=1.01)

        assert result.status == CacheStatus.MISS
