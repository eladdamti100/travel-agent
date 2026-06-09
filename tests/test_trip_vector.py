"""Tests for the multi-dimensional travel vector space (missions 1.4 / 1.5).

Mission 2.1 changed build_trip_vector to return a pure 384-d text embedding.
Structured dimensions (travel_style, food, etc.) are no longer concatenated
into the vector — they are enforced by SQLite pre-filtering instead.  Tests
that rely on the old 411-d hybrid behavior are marked xfail.
"""

import numpy as np
import pytest

from src.services.trip_vector import (
    BUDGET_TIER_MAX_DIFF,
    COVERAGE_THRESHOLD_CAP,
    DIMENSION_WEIGHTS,
    DURATION_MAX_BUCKET_DIFF,
    _STRUCTURED_ENCODERS,
    _one_hot,
    adjusted_threshold,
    build_trip_vector,
    check_hard_filters,
    compute_coverage,
)

_EMB = [1.0] + [0.0] * 383   # unit text embedding


def _v(ctx=None):
    vec, _ = build_trip_vector(_EMB, ctx or {})
    return vec


def _cov(ctx):
    return compute_coverage(ctx)


# ── Vector properties ────────────────────────────────────────────────────────

def test_vector_shape_dtype_and_unit_norm():
    # Mission 2.1: vector is now 384-d (pure text embedding), not 411-d.
    vec, cov = build_trip_vector(_EMB, {})
    assert vec.shape == (384,)
    assert vec.dtype == np.float32
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
    assert isinstance(cov, float)

def test_weights_sum_to_one():
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6

def test_encoder_keys_in_dimension_weights():
    for key, _ in _STRUCTURED_ENCODERS:
        assert key in DIMENSION_WEIGHTS

def test_same_context_identical_vectors():
    ctx = {"travel_style": "family", "num_travelers": 4, "travel_month": "july"}
    assert np.allclose(_v(ctx), _v(ctx))

def test_missing_context_zeros_structured_dims():
    # Mission 2.1: coverage is still computed; vector is 384-d with no tail.
    vec, cov = build_trip_vector(_EMB, {})
    assert cov == 0.0
    assert vec.shape == (384,)

@pytest.mark.xfail(reason="Mission 2.1: structured dims removed from vector; "
                           "food difference enforced by SQLite pre-filter now.")
def test_food_none_different_from_explicit_none():
    """food=None (missing) must differ from food='none' (no restriction)."""
    assert not np.allclose(_v({}), _v({"food_preference": "none"}))


# ── Structured dimensions differentiate ─────────────────────────────────────

@pytest.mark.xfail(reason="Mission 2.1: structured dims removed from vector.")
@pytest.mark.parametrize("field,a,b", [
    ("travel_style",  "budget",   "luxury"),
    ("num_travelers", 1,          4),
    ("travel_month",  "july",     "january"),
    ("food_preference", "vegan",  "kosher"),
])
def test_different_values_produce_different_vectors(field, a, b):
    assert not np.allclose(_v({field: a}), _v({field: b}))


@pytest.mark.parametrize("abbrev,full", [
    ("jul", "july"), ("jan", "january"), ("dec", "december"),
    ("sep", "september"), ("mar", "march"),
])
def test_abbreviated_month_same_as_full(abbrev, full):
    assert np.allclose(_v({"travel_month": abbrev}), _v({"travel_month": full}))


# ── Edge cases ───────────────────────────────────────────────────────────────

@pytest.mark.xfail(reason="Mission 2.1: vector is now pure text embedding — "
                           "structured fields no longer lower similarity.")
def test_full_vs_empty_similarity_below_threshold():
    full_ctx = {"travel_style": "luxury", "num_travelers": 2, "activity_preference": "museums",
                "travel_month": "july", "food_preference": "none", "flight_preference": "business"}
    assert float(np.dot(_v(full_ctx), _v({}))) < 0.85

@pytest.mark.xfail(reason="Mission 2.1: structured dims removed from vector.")
def test_complementary_fields_lower_similarity_than_matching():
    v_style = _v({"travel_style": "luxury"})
    assert float(np.dot(v_style, _v({"travel_style": "luxury"}))) > \
           float(np.dot(v_style, _v({"food_preference": "vegan"})))

def test_word_boundary_no_false_match():
    result = _one_hot(["nature", "nightlife", "museums"], "no")
    assert np.all(result == 0.0)

def test_word_in_phrase_matches():
    result = _one_hot(["budget", "luxury", "family"], "luxury hotel")
    assert result[1] == 1.0


# ── Coverage & threshold ─────────────────────────────────────────────────────

@pytest.mark.parametrize("ctx,expected", [
    ({}, 0.0),
    ({"travel_style": "luxury", "num_travelers": 2,
      "activity_preference": "museums", "travel_month": "july",
      "food_preference": "none", "flight_preference": "business"}, 1.0),
    ({"travel_style": "luxury", "num_travelers": 2}, pytest.approx(2/6)),
])
def test_coverage_values(ctx, expected):
    assert _cov(ctx) == expected

def test_adjusted_threshold_increases_with_lower_coverage():
    assert adjusted_threshold(0.85, 1.0) <= adjusted_threshold(0.85, 0.5) <= adjusted_threshold(0.85, 0.0)

def test_adjusted_threshold_capped_and_unchanged_at_full():
    assert adjusted_threshold(0.95, 0.0) <= COVERAGE_THRESHOLD_CAP
    assert adjusted_threshold(0.85, 1.0) == pytest.approx(0.85)


# ── Hard filters ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ctx_a,ctx_b,should_pass", [
    ({"total_budget": 2000}, {"total_budget": 2200},  True),   # same bucket
    ({"total_budget": 2000}, {"total_budget": 2250},  True),   # 1 bucket diff = OK
    ({"total_budget": 500},  {"total_budget": 3000},  False),  # too far apart
    ({"total_budget": 2000}, {"total_budget": 2500},  False),  # 2 buckets = reject
    ({"duration_days": 5},   {"duration_days": 7},    True),   # both "week"
    ({"duration_days": 2},   {"duration_days": 20},   False),  # short vs long
    ({"total_budget": 2000}, {},                       True),   # missing = no filter
    ({},                     {"duration_days": 7},    True),   # missing = no filter
])
def test_hard_filters(ctx_a, ctx_b, should_pass):
    passes, _ = check_hard_filters(ctx_a, ctx_b)
    assert passes == should_pass
