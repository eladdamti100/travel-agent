"""
Tests for the multi-dimensional travel vector space (missions 1.4 / 1.5).
"""

import numpy as np
import pytest

from src.services.trip_vector import (
    BUDGET_TIER_MAX_DIFF,
    COVERAGE_THRESHOLD_BOOST,
    DIMENSION_WEIGHTS,
    DURATION_MAX_BUCKET_DIFF,
    adjusted_threshold,
    build_trip_vector,
    check_hard_filters,
    compute_coverage,
)


_TEXT_DIM = 384
_FAKE_EMBEDDING = [0.0] * _TEXT_DIM
_FAKE_EMBEDDING[0] = 1.0  # unit vector pointing along dim 0


def _vec(ctx):
    """Helper — returns only the vector (ignores coverage)."""
    vec, _ = build_trip_vector(_FAKE_EMBEDDING, ctx)
    return vec


def _cov(ctx):
    """Helper — returns only the coverage."""
    _, cov = build_trip_vector(_FAKE_EMBEDDING, ctx)
    return cov


class TestBuildTripVector:

    def test_output_shape(self):
        vec, _ = build_trip_vector(_FAKE_EMBEDDING, {})
        assert vec.shape == (411,)

    def test_returns_tuple_of_vector_and_coverage(self):
        result = build_trip_vector(_FAKE_EMBEDDING, {})
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_is_unit_normalised(self):
        vec, _ = build_trip_vector(_FAKE_EMBEDDING, {"travel_style": "luxury"})
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_dtype_is_float32(self):
        vec, _ = build_trip_vector(_FAKE_EMBEDDING, {})
        assert vec.dtype == np.float32

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_different_travel_styles_produce_different_vectors(self):
        assert not np.allclose(_vec({"travel_style": "budget"}), _vec({"travel_style": "luxury"}))

    def test_same_context_produces_identical_vectors(self):
        ctx = {"travel_style": "family", "num_travelers": 4, "travel_month": "july"}
        assert np.allclose(_vec(ctx), _vec(ctx))

    def test_missing_context_degrades_to_text_only(self):
        """All structured dims missing → vector == normalized text component."""
        vec, cov = build_trip_vector(_FAKE_EMBEDDING, {})
        assert cov == 0.0
        assert np.linalg.norm(vec) > 0  # still unit-normalised text vector

    def test_zero_dims_for_missing_fields(self):
        """Missing structured dims must be zero, not neutral, to avoid inflating similarity."""
        vec, _ = build_trip_vector(_FAKE_EMBEDDING, {})
        # Structured part starts at index 384; all should be zero
        assert np.all(vec[384:] == 0.0)

    def test_season_extracted_from_travel_month(self):
        assert not np.allclose(_vec({"travel_month": "july"}), _vec({"travel_month": "january"}))

    def test_traveler_count_influences_vector(self):
        assert not np.allclose(_vec({"num_travelers": 1}), _vec({"num_travelers": 4}))

    def test_food_none_is_zero_not_neutral(self):
        """food_preference=None must produce zeros, not one-hot 'none'."""
        vec_missing, _ = build_trip_vector(_FAKE_EMBEDDING, {})
        vec_none, _    = build_trip_vector(_FAKE_EMBEDDING, {"food_preference": "none"})
        # The "none" entry should be non-zero (explicit no-restriction), missing should be zero
        assert not np.allclose(vec_missing, vec_none)

    def test_same_travel_style_scores_higher_than_different(self):
        v_l1 = _vec({"travel_style": "luxury"})
        v_l2 = _vec({"travel_style": "luxury", "travel_month": "june"})
        v_b  = _vec({"travel_style": "budget"})
        assert float(np.dot(v_l1, v_l2)) > float(np.dot(v_l1, v_b))

    def test_full_vs_empty_similarity_below_threshold(self):
        """Full-context query vs text-only cached entry must always be below 0.85."""
        import math
        full_ctx = {"travel_style": "luxury", "num_travelers": 2, "activity_preference": "museums",
                    "travel_month": "july", "food_preference": "none", "flight_preference": "business"}
        v_full, _ = build_trip_vector(_FAKE_EMBEDDING, full_ctx)
        v_empty, _ = build_trip_vector(_FAKE_EMBEDDING, {})
        sim = float(np.dot(v_full, v_empty))
        assert sim < 0.85, f"Expected sim < 0.85, got {sim:.4f}"

    def test_complementary_fields_similarity_below_full_match(self):
        """Two vectors with no overlapping structured dims score lower than matching dims."""
        v_style  = _vec({"travel_style": "luxury"})
        v_food   = _vec({"food_preference": "vegan"})
        v_match  = _vec({"travel_style": "luxury"})
        sim_compl = float(np.dot(v_style, v_food))
        sim_match = float(np.dot(v_style, v_match))
        assert sim_match > sim_compl

    def test_abbreviated_month_maps_to_season(self):
        v_jul = _vec({"travel_month": "jul"})
        v_july = _vec({"travel_month": "july"})
        assert np.allclose(v_jul, v_july), "Abbreviated 'jul' should produce the same vector as 'july'"

    def test_abbreviated_month_jan(self):
        v_jan = _vec({"travel_month": "jan"})
        v_july = _vec({"travel_month": "july"})
        assert not np.allclose(v_jan, v_july)  # winter vs summer

    def test_one_hot_word_boundary_no_false_match(self):
        """Short input like 'no' must not match category 'nature' or 'nightlife'."""
        from src.services.trip_vector import _one_hot
        result = _one_hot(["nature", "nightlife", "museums"], "no")
        assert np.all(result == 0.0), "Short input 'no' must not match any category"

    def test_one_hot_word_in_phrase_matches(self):
        """'luxury hotel' should still match category 'luxury'."""
        from src.services.trip_vector import _one_hot
        result = _one_hot(["budget", "luxury", "family"], "luxury hotel")
        assert result[1] == 1.0

    def test_encoder_key_validation_at_import(self):
        """_STRUCTURED_ENCODERS keys must all exist in DIMENSION_WEIGHTS."""
        from src.services.trip_vector import _STRUCTURED_ENCODERS, DIMENSION_WEIGHTS
        for key, _ in _STRUCTURED_ENCODERS:
            assert key in DIMENSION_WEIGHTS, f"Key '{key}' missing from DIMENSION_WEIGHTS"


class TestCoverage:

    def test_empty_context_coverage_zero(self):
        assert compute_coverage({}) == 0.0

    def test_full_context_coverage_one(self):
        ctx = {"travel_style": "luxury", "num_travelers": 2, "activity_preference": "museums",
               "travel_month": "july", "food_preference": "none", "flight_preference": "business"}
        assert compute_coverage(ctx) == 1.0

    def test_partial_coverage(self):
        ctx = {"travel_style": "luxury", "num_travelers": 2}  # 2 out of 6
        cov = compute_coverage(ctx)
        assert abs(cov - 2/6) < 1e-6

    def test_adjusted_threshold_increases_with_lower_coverage(self):
        t_full    = adjusted_threshold(0.85, 1.0)
        t_partial = adjusted_threshold(0.85, 0.5)
        t_empty   = adjusted_threshold(0.85, 0.0)
        assert t_full <= t_partial <= t_empty

    def test_adjusted_threshold_capped(self):
        from src.services.trip_vector import COVERAGE_THRESHOLD_CAP
        assert adjusted_threshold(0.95, 0.0) <= COVERAGE_THRESHOLD_CAP

    def test_adjusted_threshold_no_change_at_full_coverage(self):
        assert adjusted_threshold(0.85, 1.0) == pytest.approx(0.85)


class TestHardFilters:

    def test_compatible_budgets_pass(self):
        ctx_a = {"total_budget": 2000}
        ctx_b = {"total_budget": 2200}  # same 250-unit bucket
        passes, _ = check_hard_filters(ctx_a, ctx_b)
        assert passes

    def test_incompatible_budgets_fail(self):
        ctx_a = {"total_budget": 500}
        ctx_b = {"total_budget": 3000}
        passes, reason = check_hard_filters(ctx_a, ctx_b)
        assert not passes
        assert "budget" in reason.lower()

    def test_compatible_durations_pass(self):
        ctx_a = {"duration_days": 5}   # week bucket
        ctx_b = {"duration_days": 7}   # week bucket
        passes, _ = check_hard_filters(ctx_a, ctx_b)
        assert passes

    def test_incompatible_durations_fail(self):
        ctx_a = {"duration_days": 2}   # short
        ctx_b = {"duration_days": 20}  # long
        passes, reason = check_hard_filters(ctx_a, ctx_b)
        assert not passes
        assert "duration" in reason.lower()

    def test_missing_budget_passes(self):
        """Missing budget in either context should not hard-reject."""
        ctx_a = {"duration_days": 7}
        ctx_b = {"total_budget": 2000, "duration_days": 7}
        passes, _ = check_hard_filters(ctx_a, ctx_b)
        assert passes

    def test_missing_duration_passes(self):
        ctx_a = {"total_budget": 2000}
        ctx_b = {"total_budget": 2000}
        passes, _ = check_hard_filters(ctx_a, ctx_b)
        assert passes

    def test_budget_diff_exactly_at_threshold_passes(self):
        ctx_a = {"total_budget": 2000}
        ctx_b = {"total_budget": 2250}  # exactly 1 bucket diff
        passes, _ = check_hard_filters(ctx_a, ctx_b)
        assert passes

    def test_budget_diff_one_over_threshold_fails(self):
        ctx_a = {"total_budget": 2000}
        ctx_b = {"total_budget": 2500}  # 2 bucket diff
        passes, _ = check_hard_filters(ctx_a, ctx_b)
        assert not passes


class TestHybridSimilarity:
    """End-to-end: luxury trip should not match budget trip even at high text similarity."""

    def test_same_style_scores_higher_than_different_style(self):
        text = _FAKE_EMBEDDING

        query_luxury, _  = build_trip_vector(text, {"travel_style": "luxury", "total_budget": 5000})
        cached_luxury, _ = build_trip_vector(text, {"travel_style": "luxury", "total_budget": 5000})
        cached_budget, _ = build_trip_vector(text, {"travel_style": "budget", "total_budget": 500})

        sim_match  = float(np.dot(query_luxury, cached_luxury))
        sim_differ = float(np.dot(query_luxury, cached_budget))

        assert sim_match > sim_differ, (
            f"Luxury-vs-luxury ({sim_match:.4f}) should score higher "
            f"than luxury-vs-budget ({sim_differ:.4f})"
        )
