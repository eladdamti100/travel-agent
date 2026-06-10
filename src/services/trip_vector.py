"""
Multi-dimensional travel vector space for smarter cache similarity.

Architecture: hybrid approach
  Layer 1 — Hard filters: budget tier and duration bucket must be compatible.
             A mismatch forces MISS regardless of text similarity score.
  Layer 2 — Weighted soft vector: weighted concatenation of text embedding
             and structured dimensions extracted from TripContext.

Dimensions and weights (all sourced from TripContext — no extra extraction needed):

  text_embedding     (384d)  60%  primary semantic signal
  travel_style         (5d)  15%  budget/luxury/family/adventure/relaxed
  traveler_type        (4d)  10%  solo/couple/small_group/large_group
  activity_preference  (6d)   8%  museums/nature/nightlife/shopping/kids/wellness
  season               (4d)   5%  spring/summer/autumn/winter (from travel_month)
  food_preference      (5d)   1%  none/vegan/kosher/halal/vegetarian
  flight_tier          (3d)   1%  economy/business/direct

Total vector: 411 dimensions, unit-normalised float32.

Extension: add a new dimension by appending an entry to DIMENSION_WEIGHTS
and implementing an encoder. Zero changes elsewhere.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Max threshold boost applied when coverage = 0 (all structured dims missing).
# coverage=1.0 → +0.00  coverage=0.5 → +0.025  coverage=0.0 → +0.05
COVERAGE_THRESHOLD_BOOST = 0.05
COVERAGE_THRESHOLD_CAP   = 0.95  # never require > this regardless of coverage

import numpy as np

# ── Weights ───────────────────────────────────────────────────────────────────
# Each value is the fraction of influence in the final cosine similarity.
# Component vectors are scaled by sqrt(weight) before concatenation so that
# after L2 normalisation the dot product equals the weighted cosine similarity.

DIMENSION_WEIGHTS: Dict[str, float] = {
    "text":          0.60,
    "travel_style":  0.15,
    "traveler_type": 0.10,
    "activity":      0.08,
    "season":        0.05,
    "food":          0.01,
    "flight_tier":   0.01,
}

# Validate at import time — catch mis-edits immediately rather than silently
# producing a corrupted vector space.
assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6, (
    f"DIMENSION_WEIGHTS must sum to 1.0, got {sum(DIMENSION_WEIGHTS.values()):.6f}. "
    "Adjust the weights before importing this module."
)

# ── Duration normaliser (shared with semantic_cache.build_trip_cache_key) ────
# Lives here to avoid a circular import: semantic_cache imports trip_vector,
# so trip_vector must not import semantic_cache at module level.

def normalize_duration_bucket(v: Any) -> Optional[str]:
    """
    Buckets trip duration so nearby durations share a cache entry.
    1–3 days → short | 4–7 → week | 8–14 → extended | 15+ → long
    """
    try:
        days = int(v)
    except (TypeError, ValueError):
        return None
    if days <= 0:
        return None
    if days <= 3:
        return "short"
    if days <= 7:
        return "week"
    if days <= 14:
        return "extended"
    return "long"


# ── Hard filter thresholds ────────────────────────────────────────────────────

# Max allowed budget tier difference (250-unit buckets) before hard rejection.
BUDGET_TIER_MAX_DIFF = 1   # $2000 vs $2250 → OK; $2000 vs $3000 → MISS

# Duration buckets in ascending order of length.
_DURATION_ORDER = ["short", "week", "extended", "long"]
DURATION_MAX_BUCKET_DIFF = 1  # short↔week → OK; short↔long → MISS

# ── Category vocabularies ─────────────────────────────────────────────────────

_TRAVEL_STYLE_CATS = ["budget", "luxury", "family", "adventure", "relaxed"]
_TRAVELER_CATS     = ["solo", "couple", "small_group", "large_group"]
_ACTIVITY_CATS     = ["museums", "nature", "nightlife", "shopping", "kids", "wellness"]
_SEASON_CATS       = ["spring", "summer", "autumn", "winter"]
_FOOD_CATS         = ["none", "vegan", "kosher", "halal", "vegetarian"]
_FLIGHT_CATS       = ["economy", "business", "direct"]

_MONTH_TO_SEASON: Dict[str, str] = {
    # Full names
    "december": "winter", "january": "winter", "february": "winter",
    "march":    "spring", "april":   "spring", "may":      "spring",
    "june":     "summer", "july":    "summer", "august":   "summer",
    "september":"autumn", "october": "autumn", "november": "autumn",
    # 3-letter abbreviations (LLM output varies)
    "dec": "winter", "jan": "winter", "feb": "winter",
    "mar": "spring", "apr": "spring",
    "jun": "summer", "jul": "summer", "aug": "summer",
    "sep": "autumn", "oct": "autumn", "nov": "autumn",
}

_TRAVELER_BUCKET: Dict[int, str] = {
    1: "solo", 2: "couple", 3: "small_group", 4: "small_group",
}


# ── Encoding helpers ──────────────────────────────────────────────────────────
#
# Missing dimensions return ZERO vectors — not neutral uniform vectors.
# Zero means "no information — ignore this dimension in similarity scoring."
# Neutral (1/n each) would cause missing×missing to produce a non-zero dot
# product, artificially inflating similarity between two sparse vectors.

def _one_hot(categories: List[str], value: Optional[str]) -> np.ndarray:
    """
    One-hot encodes a known categorical value.
    Returns zeros when value is None or unrecognised (dimension absent).

    Matching uses word-boundary regex so short inputs like "no" don't
    accidentally match category words like "nature" or "nightlife".
    """
    if not value:
        return np.zeros(len(categories), dtype=np.float32)
    v = value.strip().lower()
    vec = np.zeros(len(categories), dtype=np.float32)
    for i, cat in enumerate(categories):
        # Match if the category word appears as a whole word in v, or v exactly equals cat.
        if v == cat or re.search(r'\b' + re.escape(cat) + r'\b', v):
            vec[i] = 1.0
            return vec
    return np.zeros(len(categories), dtype=np.float32)


def _multi_hot(categories: List[str], value: Optional[str]) -> np.ndarray:
    """
    Multi-hot encodes a field that may contain several values.
    Returns zeros when value is None or nothing matches (dimension absent).
    L2-normalises so the vector has unit magnitude when any category is present.
    """
    if not value:
        return np.zeros(len(categories), dtype=np.float32)
    v = value.strip().lower()
    vec = np.zeros(len(categories), dtype=np.float32)
    found = False
    for i, cat in enumerate(categories):
        if cat in v:
            vec[i] = 1.0
            found = True
    if not found:
        return np.zeros(len(categories), dtype=np.float32)
    # Normalize by L2 norm (not sum) so multi-hot vectors have the same
    # unit magnitude as one-hot vectors — prevents multiple-preference
    # users from having less similarity influence than single-preference ones.
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Per-dimension encoders ────────────────────────────────────────────────────

def _enc_travel_style(ctx: dict) -> np.ndarray:
    return _one_hot(_TRAVEL_STYLE_CATS, ctx.get("travel_style"))


def _enc_traveler_type(ctx: dict) -> np.ndarray:
    n = ctx.get("num_travelers")
    if n is not None:
        try:
            bucket = _TRAVELER_BUCKET.get(int(n), "large_group")
            return _one_hot(_TRAVELER_CATS, bucket)
        except (TypeError, ValueError):
            pass
    return np.zeros(len(_TRAVELER_CATS), dtype=np.float32)  # unknown → absent


def _enc_activity(ctx: dict) -> np.ndarray:
    return _multi_hot(_ACTIVITY_CATS, ctx.get("activity_preference"))


def _enc_season(ctx: dict) -> np.ndarray:
    month = (ctx.get("travel_month") or "").strip().lower()
    season = _MONTH_TO_SEASON.get(month)
    return _one_hot(_SEASON_CATS, season)  # returns zeros if month unknown


def _enc_food(ctx: dict) -> np.ndarray:
    food = ctx.get("food_preference")
    # None / "" → missing (zero) — NOT the same as "none" (no restrictions).
    # A user who explicitly said "no restrictions" encodes as one-hot "none".
    if not food:
        return np.zeros(len(_FOOD_CATS), dtype=np.float32)
    return _one_hot(_FOOD_CATS, food)


def _enc_flight_tier(ctx: dict) -> np.ndarray:
    return _one_hot(_FLIGHT_CATS, ctx.get("flight_preference"))


# ── Coverage computation ──────────────────────────────────────────────────────

# Ordered list of (weight_key, encoder) for structured (non-text) dimensions.
# Used to compute coverage and build the structured part of the vector.
_STRUCTURED_ENCODERS = [
    ("travel_style",  _enc_travel_style),
    ("traveler_type", _enc_traveler_type),
    ("activity",      _enc_activity),
    ("season",        _enc_season),
    ("food",          _enc_food),
    ("flight_tier",   _enc_flight_tier),
]

# Validate at import time that every encoder key has a corresponding weight.
# Catches the easy mistake of adding an encoder but forgetting its weight.
_missing_weights = [k for k, _ in _STRUCTURED_ENCODERS if k not in DIMENSION_WEIGHTS]
assert not _missing_weights, (
    f"_STRUCTURED_ENCODERS keys missing from DIMENSION_WEIGHTS: {_missing_weights}. "
    "Add the missing weights before importing this module."
)


def compute_coverage(trip_context: dict) -> float:
    """
    Returns the fraction of structured dimensions that are present (non-zero).

    0.0 = no structured info (pure text similarity will be used)
    1.0 = all structured dimensions are known
    """
    present = sum(
        1 for _, enc in _STRUCTURED_ENCODERS
        if np.any(enc(trip_context) != 0)
    )
    return present / len(_STRUCTURED_ENCODERS)


def adjusted_threshold(base_threshold: float, coverage: float) -> float:
    """
    Returns a coverage-adjusted similarity threshold.

    Low coverage → higher required similarity (less info → less trust).
      coverage=1.0 → base_threshold + 0.00
      coverage=0.5 → base_threshold + 0.025
      coverage=0.0 → base_threshold + 0.05 (capped at COVERAGE_THRESHOLD_CAP)
    """
    boost = (1.0 - coverage) * COVERAGE_THRESHOLD_BOOST
    return min(base_threshold + boost, COVERAGE_THRESHOLD_CAP)


# ── Main builder ──────────────────────────────────────────────────────────────

def build_trip_vector(
    text_embedding: List[float],
    trip_context: dict,
) -> Tuple[np.ndarray, float]:
    """
    Returns the unit-normalised text embedding (384d) and structured coverage.

    The structured dimensions are no longer concatenated into the vector —
    business constraints (destination, budget) are enforced by SQLite
    pre-filtering before cosine similarity runs.  Coverage is still computed
    so adjusted_threshold() can raise the bar when context is sparse.
    """
    struct_vecs = [enc(trip_context) for _, enc in _STRUCTURED_ENCODERS]
    coverage = sum(1 for v in struct_vecs if np.any(v != 0)) / len(struct_vecs)

    vec = np.array(text_embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm

    return vec, coverage


# ── Hard filters ──────────────────────────────────────────────────────────────

def _budget_tier_diff(ctx_a: dict, ctx_b: dict) -> Optional[int]:
    b_a, b_b = ctx_a.get("total_budget"), ctx_b.get("total_budget")
    if b_a is None or b_b is None:
        return None
    try:
        return abs(round(float(b_a) / 250) - round(float(b_b) / 250))
    except (TypeError, ValueError):
        return None


def _duration_bucket_diff(ctx_a: dict, ctx_b: dict) -> Optional[int]:
    d_a = normalize_duration_bucket(ctx_a.get("duration_days"))
    d_b = normalize_duration_bucket(ctx_b.get("duration_days"))
    if d_a is None or d_b is None:
        return None
    try:
        return abs(_DURATION_ORDER.index(d_a) - _DURATION_ORDER.index(d_b))
    except ValueError:
        return None


def check_hard_filters(
    query_context: dict,
    cached_context: dict,
) -> Tuple[bool, str]:
    """
    Checks whether two trip contexts are compatible for cache reuse.

    Returns (passes: bool, reason: str).
    When passes=False the cached entry must not be served regardless of
    the vector similarity score.
    """
    budget_diff = _budget_tier_diff(query_context, cached_context)
    if budget_diff is not None and budget_diff > BUDGET_TIER_MAX_DIFF:
        return False, (
            f"Budget tier gap ({budget_diff} buckets) exceeds "
            f"max allowed ({BUDGET_TIER_MAX_DIFF})."
        )

    dur_diff = _duration_bucket_diff(query_context, cached_context)
    if dur_diff is not None and dur_diff > DURATION_MAX_BUCKET_DIFF:
        return False, (
            f"Duration bucket gap ({dur_diff}) exceeds "
            f"max allowed ({DURATION_MAX_BUCKET_DIFF})."
        )

    q_origin = (query_context.get("origin_airport") or "").strip().upper()
    c_origin = (cached_context.get("origin_airport") or "").strip().upper()
    if q_origin and c_origin and q_origin != c_origin:
        return False, f"Origin mismatch: {q_origin} vs {c_origin}."

    return True, "Compatible."
