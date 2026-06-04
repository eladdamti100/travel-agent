"""
Semantic cache service.

Stores and retrieves previous trip-planning answers using local embeddings.

Embedding model:
    sentence-transformers/all-MiniLM-L6-v2

Storage:
    data/semantic_cache.db

Similarity:
    cosine similarity

Default hit threshold:
    0.85
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# HuggingFace / torch noise vars are set once in run.py before any import.
# Suppress library loggers directly (safe to do at import time — no env mutation).
import logging as _logging
_logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.cache import CacheCheckResult, CacheEntry, CacheStatus
from src.utils.logger import get_logger

logger = get_logger("semantic_cache")

# Use ~/.cache to avoid iCloud Drive file-coordination interference with WAL mode.
_CACHE_DB_PATH = Path.home() / ".cache" / "travel-agent" / "semantic_cache.db"
_CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HIT_THRESHOLD = 0.85
_MAX_ROWS_PER_ROUTE = 200

TTL_DAYS_DB = 30
TTL_DAYS_WEB = 3
_MAX_TTL_DAYS = 365

# Cache key versioning — bump when key format changes so old entries can be
# bulk-invalidated via invalidate_cache_by_fields({"key_version": "<old>"}).
_CACHE_KEY_VERSION = "v2"

# Open currency registry — any ISO-4217 code can be registered at runtime.
# Call register_currency("thb") to add Thai Baht, etc.
# Unknown currencies fall back to _DEFAULT_CURRENCY rather than crashing.
_REGISTERED_CURRENCIES: set = {"usd", "eur", "gbp", "ils", "jpy", "aud", "cad",
                                "chf", "cny", "inr", "brl", "mxn", "sgd", "hkd",
                                "nok", "sek", "dkk", "pln", "czk", "huf", "thb",
                                "try", "zar", "nzd", "krw", "aed", "sar"}
_DEFAULT_CURRENCY = "usd"


_CURRENCY_RE = re.compile(r"^[A-Za-z]{2,4}$")


def register_currency(currency_code: str) -> None:
    """
    Register a new currency code so it produces a distinct cache key bucket.

    currency_code should be a 2–4 letter ISO-4217 alphabetic code (e.g. "thb", "cop").
    Raises ValueError for empty or non-alphabetic codes.
    Unknown currencies default to 'usd' until registered here.
    """
    code = currency_code.strip().lower()
    if not code or not _CURRENCY_RE.match(code):
        raise ValueError(
            f"Invalid currency code {currency_code!r}. "
            "Must be 2–4 alphabetic characters (e.g. 'thb', 'usd')."
        )
    _REGISTERED_CURRENCIES.add(code)
    logger.info("Registered currency: %s", code)

_embedding_model: Optional[SentenceTransformer] = None
_embedding_model_lock = threading.Lock()
_db_initialized: bool = False
_db_init_lock = threading.Lock()

# 5% chance of running expired-entry cleanup on each lookup call.
_CLEANUP_ON_LOOKUP_PROBABILITY = 0.05

_VALID_SOURCES = ("db", "web")


def _get_embedding_model() -> SentenceTransformer:
    """
    Lazily loads and returns the local sentence-transformers embedding model.

    Thread-safe via double-checked locking — two concurrent first-callers
    will not both load the model simultaneously.
    """
    global _embedding_model

    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)

    return _embedding_model


def warm_embedding_model() -> bool:
    """
    Eagerly loads the embedding model so the first user turn is not delayed.

    Returns True when the model had to be downloaded, False when it was already
    cached locally — callers can use this to show a friendly first-run message.
    """
    model_cache = Path.home() / ".cache" / "huggingface" / "hub"
    already_cached = model_cache.exists() and any(
        "all-MiniLM-L6-v2" in p.name for p in model_cache.iterdir()
    )
    _get_embedding_model()
    return not already_cached


def initialize_cache_db() -> None:
    """
    Creates the semantic cache SQLite database and table if they do not exist.

    Guarded by _db_initialized so schema checks run only once per process.
    """
    global _db_initialized
    if _db_initialized:
        return

    with _db_init_lock:
        if _db_initialized:  # re-check after acquiring lock (double-checked locking)
            return

        _CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(_CACHE_DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    route TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    compressed_answer TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT 'db',
                    ttl_days INTEGER NOT NULL DEFAULT 30,
                    trip_vector_json TEXT,
                    trip_context_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

            # Migrate existing databases that predate these columns.
            for migration in (
                "ALTER TABLE semantic_cache ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
                "ALTER TABLE semantic_cache ADD COLUMN compressed_answer TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE semantic_cache ADD COLUMN source TEXT NOT NULL DEFAULT 'db'",
                "ALTER TABLE semantic_cache ADD COLUMN ttl_days INTEGER NOT NULL DEFAULT 30",
                "ALTER TABLE semantic_cache ADD COLUMN trip_vector_json TEXT",
                "ALTER TABLE semantic_cache ADD COLUMN trip_context_json TEXT",
            ):
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # column already exists

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_route
                ON semantic_cache(route)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_route_created
                ON semantic_cache(route, created_at)
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_route_query
                ON semantic_cache(route, normalized_query)
                """
            )

            conn.commit()

        # Purge entries built with an old key version — they can never be hit
        # by exact match (wrong version prefix) and are dead weight in the DB.
        with sqlite3.connect(_CACHE_DB_PATH) as conn:
            deleted = conn.execute(
                """
                DELETE FROM semantic_cache
                WHERE normalized_query LIKE 'key_version:%'
                  AND normalized_query NOT LIKE ?
                """,
                (f"key_version:{_CACHE_KEY_VERSION}%",),
            ).rowcount
            conn.commit()
        if deleted:
            logger.info("Purged %d stale cache entries with outdated key version.", deleted)

        _db_initialized = True


def normalize_query(query: str) -> str:
    """
    Normalizes a user query before embedding and comparison.

    This keeps semantic meaning while reducing noise from casing,
    extra whitespace, and punctuation differences.
    """
    normalized = query.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[“”]", '"', normalized)
    normalized = re.sub(r"[‘’]", "'", normalized)

    return normalized


def embed_text(text: str) -> List[float]:
    """
    Creates an embedding vector for the given text.
    """
    model = _get_embedding_model()
    vector = model.encode(text, normalize_embeddings=True)

    return vector.astype(float).tolist()


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Computes cosine similarity between two vectors.

    Embeddings are already normalized, but this function remains defensive
    in case future embedding providers return non-normalized vectors.
    """
    a = np.array(vector_a, dtype=float)
    b = np.array(vector_b, dtype=float)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def _load_embedding_index(route: str) -> List[Dict[str, Any]]:
    """
    Loads only id, query, and embedding_json for all rows in a route.

    Intentionally excludes the large answer/compressed_answer columns so the
    similarity scan never loads megabytes of cached text into memory.
    The full row is fetched separately only when a hit is confirmed.
    """
    initialize_cache_db()

    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, query, embedding_json, trip_vector_json, trip_context_json
            FROM semantic_cache
            WHERE route = ?
              AND created_at >= datetime('now', '-' || ttl_days || ' days')
            ORDER BY id DESC
            """,
            (route,),
        ).fetchall()

    return [dict(row) for row in rows]


def _fetch_row_by_id(row_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetches the full cache row (including answer) for a single id.
    """
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT query, answer, compressed_answer, source, ttl_days
            FROM semantic_cache
            WHERE id = ?
            """,
            (row_id,),
        ).fetchone()

    return dict(row) if row else None

def find_exact_cached_answer(
    query: str,
    *,
    route: str = "cache_check",
) -> CacheCheckResult:
    """
    Finds an exact cache match by normalized_query before semantic similarity is used.
    """
    initialize_cache_db()

    normalized_query = normalize_query(query)

    logger.info(
        "Exact cache lookup. route=%s normalized_query=%s",
        route,
        normalized_query,
    )

    # Single connection — fetch the latest row regardless of TTL, then check
    # validity in Python. Avoids a second DB round-trip on miss.
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT query, answer, compressed_answer, source, ttl_days,
                   CASE WHEN created_at >= datetime('now', '-' || ttl_days || ' days')
                        THEN 1 ELSE 0 END AS is_valid
            FROM semantic_cache
            WHERE route = ? AND normalized_query = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (route, normalized_query),
        ).fetchone()

    if not row:
        miss_reason = "No cache entry found for this query."
    elif not row["is_valid"]:
        miss_reason = "Cache entry expired (TTL exceeded)."
        if row["source"] == "web":
            with sqlite3.connect(_CACHE_DB_PATH) as del_conn:
                del_conn.execute(
                    "DELETE FROM semantic_cache WHERE route = ? AND normalized_query = ?",
                    (route, normalized_query),
                )
            logger.info(
                "Deleted expired web cache entry immediately. route=%s query=%s",
                route,
                normalized_query,
            )
        row = None  # treat as miss

    if row is None:
        logger.info(
            "Exact cache MISS. route=%s normalized_query=%s reason=%s",
            route,
            normalized_query,
            miss_reason,
        )

        return CacheCheckResult(
            status=CacheStatus.MISS,
            similarity_score=0.0,
            matched_query=None,
            cached_answer=None,
            cached_compressed_answer=None,
            reason=miss_reason,
        )

    logger.info(
        "Exact cache HIT. route=%s matched_query=%s source=%s ttl_days=%s",
        route,
        row["query"],
        row["source"],
        row["ttl_days"],
    )

    return CacheCheckResult(
        status=CacheStatus.HIT,
        similarity_score=1.0,
        matched_query=row["query"],
        cached_answer=row["answer"],
        cached_compressed_answer=row["compressed_answer"] or None,
        reason="Found an exact structured cache match.",
        source=row["source"],
        ttl_days=row["ttl_days"],
    )

def find_cached_answer(
    query: str,
    *,
    route: str = "cache_check",
    threshold: float = DEFAULT_HIT_THRESHOLD,
    trip_context: Optional[Dict[str, Any]] = None,
) -> CacheCheckResult:
    """
    Finds the best semantic cache match for the given query.

    Two-phase lookup:
      1. Load only id + embedding_json for all rows — no large answer text.
      2. Fetch the full answer only for the single best-matching row.

    Returns HIT if the best cosine similarity is at least threshold.
    Otherwise returns MISS.
    """
    initialize_cache_db()

    # Always purge expired web entries before searching — live data must never
    # surface as a semantic hit after its TTL elapses.
    _purge_expired_web_entries(route=route)

    # Probabilistic background cleanup for general (non-web) housekeeping.
    if random.random() < _CLEANUP_ON_LOOKUP_PROBABILITY:
        _cleanup_cache(route=route)

    exact_result = find_exact_cached_answer(query=query, route=route)
    if exact_result.status == CacheStatus.HIT:
        return exact_result

    normalized_query = normalize_query(query)

    # Structured trip keys (identified by the key_version: prefix) must match
    # exactly — semantic fuzzy matching would give false positives because two
    # structured keys for different cities look textually similar.
    if f"key_version:{_CACHE_KEY_VERSION}" in normalized_query:
        return CacheCheckResult(
            status=CacheStatus.MISS,
            similarity_score=0.0,
            matched_query=None,
            cached_answer=None,
            reason="Structured key requires exact match only.",
        )

    logger.info(
        "Semantic cache lookup. route=%s normalized_query=%s",
        route,
        normalized_query,
    )

    query_embedding = embed_text(normalized_query)

    # Build hybrid trip vector for the query when TripContext is available.
    # All trip_vector imports are lazy to avoid circular import at module level.
    query_trip_vec: Optional[np.ndarray] = None
    query_coverage: float = 0.0
    effective_threshold = threshold
    if trip_context:
        try:
            from src.services.trip_vector import (
                adjusted_threshold,
                build_trip_vector,
                check_hard_filters,  # imported once here; reused in hybrid path below
            )
            query_trip_vec, query_coverage = build_trip_vector(query_embedding, trip_context)
            effective_threshold = adjusted_threshold(threshold, query_coverage)
            logger.info(
                "Trip vector built. coverage=%.2f threshold %.2f→%.2f",
                query_coverage, threshold, effective_threshold,
            )
        except Exception as exc:
            logger.warning("Query trip vector build failed — using text-only: %s", exc)

    index_rows = _load_embedding_index(route=route)

    if not index_rows:
        return CacheCheckResult(
            status=CacheStatus.MISS,
            similarity_score=0.0,
            matched_query=None,
            cached_answer=None,
            reason="Semantic cache is empty for this route.",
        )

    best_score = 0.0
    best_id: Optional[int] = None
    best_query: Optional[str] = None

    # ── Hybrid similarity scan ────────────────────────────────────────────────
    # For each row:
    #   - If both query and row have trip vectors: apply hard filters first,
    #     then use hybrid vector dot product.
    #   - Otherwise: fall back to text-only batch cosine.
    # We separate rows into two groups and batch each independently.

    hybrid_rows:   List[Dict[str, Any]] = []
    textonly_rows: List[Dict[str, Any]] = []
    textonly_vecs: List[List[float]]    = []

    for row in index_rows:
        if query_trip_vec is not None and row.get("trip_vector_json"):
            hybrid_rows.append(row)
        else:
            try:
                vec = json.loads(row["embedding_json"])
                textonly_vecs.append(vec)
                textonly_rows.append(row)
            except (json.JSONDecodeError, TypeError, ValueError) as err:
                logger.warning("Skipping invalid cache embedding: %s", err)

    # -- Hybrid path --
    if hybrid_rows and query_trip_vec is not None:
        for row in hybrid_rows:
            try:
                cached_ctx = json.loads(row["trip_context_json"] or "{}")
                passes, reason = check_hard_filters(trip_context, cached_ctx)
                if not passes:
                    logger.debug("Hard filter rejected row id=%s: %s", row["id"], reason)
                    continue
                row_vec = np.array(json.loads(row["trip_vector_json"]), dtype=np.float32)
                score = float(np.dot(query_trip_vec, row_vec))
            except (json.JSONDecodeError, TypeError, ValueError) as err:
                logger.warning("Skipping invalid trip vector row id=%s: %s", row["id"], err)
                continue
            if score > best_score:
                best_score = score
                best_id = row["id"]
                best_query = row["query"]

    # -- Text-only batch path --
    if textonly_vecs:
        q_vec = np.array(query_embedding, dtype=np.float32)
        matrix = np.array(textonly_vecs, dtype=np.float32)

        if matrix.ndim == 2 and matrix.shape[1] == q_vec.shape[0]:
            scores = matrix @ q_vec
        else:
            logger.warning(
                "Embedding dimension mismatch: matrix=%s query_dim=%d — per-row fallback.",
                matrix.shape, q_vec.shape[0],
            )
            scores = np.array([
                float(np.dot(q_vec, np.array(r, dtype=np.float32)))
                if len(r) == q_vec.shape[0] else 0.0
                for r in textonly_vecs
            ])

        text_best_idx = int(np.argmax(scores))
        text_best_score = float(scores[text_best_idx])
        if text_best_score > best_score:
            best_score = text_best_score
            best_id = textonly_rows[text_best_idx]["id"]
            best_query = textonly_rows[text_best_idx]["query"]

    # Clamp to [0, 1] — floating point dot products on unit vectors can
    # produce values like 1.0000001 which would fail Pydantic's le=1.0 check.
    best_score = max(0.0, min(float(best_score), 1.0))

    if best_id is not None and best_score >= effective_threshold:
        full_row = _fetch_row_by_id(best_id)

        logger.info(
            "Semantic cache hit. score=%.4f matched_query=%s source=%s ttl_days=%s",
            best_score,
            best_query,
            full_row.get("source") if full_row else None,
            full_row.get("ttl_days") if full_row else None,
        )

        return CacheCheckResult(
            status=CacheStatus.HIT,
            similarity_score=best_score,
            matched_query=best_query,
            cached_answer=full_row["answer"] if full_row else None,
            cached_compressed_answer=(
                full_row.get("compressed_answer") or None if full_row else None
            ),
            reason="Found a sufficiently similar cached answer.",
            source=full_row.get("source") if full_row else None,
            ttl_days=full_row.get("ttl_days") if full_row else None,
        )

    logger.info("Semantic cache miss. best_score=%.4f effective_threshold=%.4f", best_score, effective_threshold)

    return CacheCheckResult(
        status=CacheStatus.MISS,
        similarity_score=best_score,
        matched_query=best_query,
        cached_answer=None,
        reason="No cached answer was similar enough.",
    )


def store_cache_entry(
    query: str,
    answer: str,
    *,
    route: str = "cache_check",
    confidence: float = 1.0,
    compressed_answer: str = "",
    source: str = "db",
    ttl_days: Optional[int] = None,
    trip_context: Optional[Dict[str, Any]] = None,
) -> CacheEntry:
    """
    Stores a new semantic cache entry.

    ttl_days is auto-derived from source when not provided:
      source="db"  → TTL_DAYS_DB (30 days)
      source="web" → TTL_DAYS_WEB (3 days)
    Pass ttl_days explicitly only to override the default policy.
    Maximum ttl_days is capped at _MAX_TTL_DAYS (365 days).

    trip_context — optional TripContext dict. When provided, a multi-dimensional
    travel vector is stored alongside the text embedding to enable hybrid
    similarity lookup with hard budget/duration filters.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"source must be one of {_VALID_SOURCES}, got {source!r}")

    if ttl_days is None:
        ttl_days = TTL_DAYS_DB if source == "db" else TTL_DAYS_WEB

    if ttl_days > _MAX_TTL_DAYS:
        logger.warning("ttl_days=%d exceeds maximum %d — capping.", ttl_days, _MAX_TTL_DAYS)
        ttl_days = _MAX_TTL_DAYS

    initialize_cache_db()

    normalized_query = normalize_query(query)
    embedding = embed_text(normalized_query)
    now = datetime.now(timezone.utc)

    # Build multi-dimensional trip vector when TripContext is available.
    trip_vector_json_str: Optional[str] = None
    trip_context_json_str: Optional[str] = None
    if trip_context:
        try:
            from src.services.trip_vector import build_trip_vector
            trip_vec, coverage = build_trip_vector(embedding, trip_context)
            trip_vector_json_str = json.dumps(trip_vec.tolist())
            trip_context_json_str = json.dumps(trip_context)
            logger.info("Trip vector stored. coverage=%.2f query=%s", coverage, query)
        except Exception as exc:
            logger.warning("Trip vector build failed — storing text-only: %s", exc)

    entry = CacheEntry(
        query=query,
        normalized_query=normalized_query,
        answer=answer,
        route=route,
        embedding=embedding,
        confidence=confidence,
        timestamp=now,
        compressed_answer=compressed_answer,
        source=source,
        ttl_days=ttl_days,
    )

    # Single connection for conflict check + insert — prevents a race condition
    # where another thread writes between two separate connection opens.
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        _existing = conn.execute(
            "SELECT source FROM semantic_cache WHERE route = ? AND normalized_query = ? ORDER BY id DESC LIMIT 1",
            (route, normalized_query),
        ).fetchone()
        if _existing and _existing["source"] != source:
            logger.warning(
                "Cache key already stored with source=%s, overwriting with source=%s. query=%s",
                _existing["source"],
                source,
                query,
            )

        # Web data is always fresh — delete any existing entries for this query
        # so the new result overwrites stale web content unconditionally.
        if source == "web" and _existing:
            conn.execute(
                "DELETE FROM semantic_cache WHERE route = ? AND normalized_query = ?",
                (route, normalized_query),
            )
            logger.info(
                "Replaced existing web cache entry with fresh data. route=%s query=%s",
                route,
                query,
            )

        conn.execute(
            """
            INSERT INTO semantic_cache (
                query,
                normalized_query,
                answer,
                route,
                embedding_json,
                confidence,
                compressed_answer,
                source,
                ttl_days,
                trip_vector_json,
                trip_context_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.query,
                entry.normalized_query,
                entry.answer,
                entry.route,
                json.dumps(entry.embedding),
                entry.confidence,
                entry.compressed_answer,
                entry.source,
                entry.ttl_days,
                trip_vector_json_str,
                trip_context_json_str,
                entry.timestamp.isoformat(),
            ),
        )

        conn.commit()

    logger.info("Cache entry committed. route=%s query=%s", route, query)

    _cleanup_cache(route=route)

    logger.info("Stored semantic cache entry for route=%s query=%s confidence=%.4f", route, query, confidence)

    return entry

# ── Cache key schema ─────────────────────────────────────────────────────────

@dataclass
class FieldDef:
    """
    Schema definition for a single cache key field.

    required  — if True and the field is missing/None, build_trip_cache_key
                returns None (the trip cannot be cached without this field).
    normalize — callable that receives the raw value and returns a clean string
                for use in the key.
    key_name  — the label used inside the cache key string.
    """
    required: bool
    normalize: Callable[[Any], Optional[str]]
    key_name: str


def _normalize_city(v: Any) -> Optional[str]:
    if not v:
        return None
    return str(v).strip().lower()


_IATA_RE = re.compile(r"^[A-Za-z]{3}$")

def _normalize_iata(v: Any) -> Optional[str]:
    if not v:
        return None
    code = str(v).strip()
    if not _IATA_RE.match(code):
        logger.warning("Invalid IATA code ignored in cache key: %r", code)
        return None
    return code.lower()


def _normalize_country(v: Any) -> Optional[str]:
    if not v:
        return None
    return str(v).strip().lower()


# Imported from trip_vector to avoid circular dependency — trip_vector must not
# import semantic_cache at module level, so the shared function lives there.
from src.services.trip_vector import normalize_duration_bucket as _normalize_duration_bucket


def _normalize_budget_with_currency(v: Any, currency: str = _DEFAULT_CURRENCY) -> Optional[str]:
    """
    Buckets budget into 250-unit bands and appends the currency code so that
    $2000 and €2000 produce distinct keys.

    Receives a (amount, currency_code) tuple pre-processed by build_trip_cache_key.
    Unknown currencies fall back to _DEFAULT_CURRENCY with a warning.
    """
    if v is None:
        return None
    try:
        amount, cur = v  # unpacked by _preprocess_budget below
        amount = float(amount)
    except (TypeError, ValueError):
        return None
    if amount <= 0:
        return None
    bucketed = round(amount / 250) * 250
    cur = cur.strip().lower() if cur else _DEFAULT_CURRENCY
    if cur not in _REGISTERED_CURRENCIES:
        logger.warning("Unregistered currency %r — defaulting to %s. Call register_currency() to add it.", cur, _DEFAULT_CURRENCY)
        cur = _DEFAULT_CURRENCY
    return f"{int(bucketed)}_{cur}"


def _normalize_traveler_bucket(v: Any) -> Optional[str]:
    """
    1 → solo | 2 → couple | 3-4 → small_group | 5+ → large_group
    """
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    if n == 1:
        return "solo"
    if n == 2:
        return "couple"
    if n <= 4:
        return "small_group"
    return "large_group"


# Ordered field registry — canonical field order ensures the key is
# deterministic regardless of the order fields are passed by the caller.
# Add new fields here; zero other code changes needed.
_CACHE_KEY_FIELDS: OrderedDict[str, FieldDef] = OrderedDict([
    ("destination_city", FieldDef(required=True,  normalize=_normalize_city,     key_name="destination_city")),
    ("duration_days",    FieldDef(required=True,  normalize=_normalize_duration_bucket, key_name="duration")),
    ("total_budget",     FieldDef(required=False, normalize=_normalize_budget_with_currency, key_name="budget_bucket")),
    ("num_travelers",    FieldDef(required=False, normalize=_normalize_traveler_bucket, key_name="travelers")),
    ("origin_airport",   FieldDef(required=False, normalize=_normalize_iata,     key_name="origin_airport")),
    ("origin_country",   FieldDef(required=False, normalize=_normalize_country,  key_name="origin_country")),
])


def register_cache_key_field(
    field_name: str,
    normalize: Callable[[Any], Optional[str]],
    *,
    key_name: Optional[str] = None,
    required: bool = False,
) -> None:
    """
    Register a new field in the cache key schema at runtime.

    Call this from any module to add a new trip dimension to the cache key
    without editing semantic_cache.py.

    Example (in Student 3's web_agent.py):
        from src.services.semantic_cache import register_cache_key_field
        register_cache_key_field("restaurant_type", lambda v: v.strip().lower(), key_name="restaurant")

    After registration, build_trip_cache_key({"restaurant_type": "vegan", ...}) will
    include restaurant:vegan in the key automatically.

    Raises ValueError if the field_name is already registered.
    """
    if field_name in _CACHE_KEY_FIELDS:
        raise ValueError(
            f"Cache key field '{field_name}' is already registered. "
            "Use a different name or update the existing entry directly."
        )
    _CACHE_KEY_FIELDS[field_name] = FieldDef(
        required=required,
        normalize=normalize,
        key_name=key_name or field_name,
    )
    logger.info("Registered new cache key field: %s (key_name=%s)", field_name, key_name or field_name)


def build_trip_cache_key(fields: Dict[str, Any]) -> Optional[str]:
    """
    Builds a deterministic, versioned, currency-aware structured cache key.

    fields — dict of trip parameters. Known keys:
        destination_city  (required)
        duration_days     (required)
        total_budget      (optional float)
        currency          (optional str, default "usd") — paired with total_budget
        num_travelers     (optional int)
        origin_airport    (optional str, IATA)
        origin_country    (optional str)

    Returns None when any required field is missing or invalid.
    Any unknown keys in fields are silently ignored, so future callers can
    pass extra fields without breaking existing behaviour.
    """
    # Pre-process budget: pair it with the currency field before normalization.
    processed = dict(fields)
    if "total_budget" in processed and processed["total_budget"] is not None:
        currency = str(processed.pop("currency", _DEFAULT_CURRENCY) or _DEFAULT_CURRENCY)
        processed["total_budget"] = (processed["total_budget"], currency)
    else:
        processed.pop("currency", None)
        processed["total_budget"] = None

    segments: List[str] = [f"key_version:{_CACHE_KEY_VERSION}"]

    for field_name, field_def in _CACHE_KEY_FIELDS.items():
        raw = processed.get(field_name)
        normalized = field_def.normalize(raw)

        if normalized is None:
            if field_def.required:
                return None  # missing required field — key cannot be built
            continue  # optional field absent — omit from key

        segments.append(f"{field_def.key_name}:{normalized}")

    return normalize_query(" | ".join(segments))


def build_trip_cache_key_from_context(
    *,
    origin_airport: Optional[str] = None,
    origin_country: Optional[str] = None,
    destination_city: Optional[str] = None,
    duration_days: Optional[int] = None,
    total_budget: Optional[float] = None,
    currency: Optional[str] = None,
    num_travelers: Optional[int] = None,
) -> Optional[str]:
    """
    Backwards-compatible shim for callers that still use keyword arguments.
    Internally delegates to build_trip_cache_key(dict).
    """
    return build_trip_cache_key({
        "destination_city": destination_city,
        "duration_days":    duration_days,
        "total_budget":     total_budget,
        "currency":         currency,
        "num_travelers":    num_travelers,
        "origin_airport":   origin_airport,
        "origin_country":   origin_country,
    })

def _purge_expired_web_entries(route: str) -> int:
    """
    Deletes all web-sourced cache entries whose TTL has elapsed for the given route.

    Called on every find_cached_answer invocation (not probabilistically) so
    expired live-data entries are removed as soon as they are no longer valid,
    rather than waiting for the background 5% cleanup.

    Returns the number of deleted rows.
    """
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        cursor = conn.execute(
            """
            DELETE FROM semantic_cache
            WHERE route = ?
              AND source = 'web'
              AND created_at < datetime('now', '-' || ttl_days || ' days')
            """,
            (route,),
        )
        deleted = cursor.rowcount

    if deleted:
        logger.info("Purged %d expired web cache entries. route=%s", deleted, route)

    return deleted


def _cleanup_cache(route: str) -> None:
    """
    Removes entries that are older than _MAX_AGE_DAYS and trims the route
    to at most _MAX_ROWS_PER_ROUTE entries (keeping the most recent ones).
    """
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.execute(
            """
            DELETE FROM semantic_cache
            WHERE route = ?
              AND created_at < datetime('now', '-' || ttl_days || ' days')
            """,
            (route,),
        )

        conn.execute(
            """
            DELETE FROM semantic_cache
            WHERE route = ?
              AND id NOT IN (
                  SELECT id FROM semantic_cache
                  WHERE route = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (route, route, _MAX_ROWS_PER_ROUTE),
        )

        conn.commit()

    logger.debug("Cache cleanup done for route=%s (max_rows=%d)", route, _MAX_ROWS_PER_ROUTE)


# ── Cache invalidation engine ────────────────────────────────────────────────

def invalidate_cache_by_fields(
    filters: Dict[str, str],
    *,
    route: str = "cache_check",
    include_freetext: bool = True,
) -> int:
    """
    Deletes all cache entries whose structured key matches every field in filters.

    filters is a dict of {field_name: value}, e.g.:
        {"destination_city": "paris"}
        {"destination_city": "paris", "duration_days": "7"}
        {"origin_airport": "tlv"}

    Any field that appears in build_trip_cache_key works here automatically —
    destination_city, origin_airport, origin_country, duration_days, budget_bucket —
    as well as any future fields added to the key format.

    When include_freetext=True, also deletes free-text (non-structured) entries
    that contain any of the filter values as plain substrings.

    Returns the total number of deleted rows.
    """
    initialize_cache_db()

    if not filters:
        logger.warning("invalidate_cache_by_fields called with empty filters — no-op.")
        return 0

    # Build LIKE conditions for structured keys: "field:value" pattern per filter.
    structured_conditions = " AND ".join(
        f"normalized_query LIKE ?"
        for _ in filters
    )
    structured_params = [
        f"%{field}:{value.strip().lower()}%"
        for field, value in filters.items()
    ]

    deleted = 0

    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        # Phase 1: structured key entries — identified by the ' | ' field separator,
        # not '%:%' which would match any colon in free-text queries.
        cursor = conn.execute(
            f"""
            DELETE FROM semantic_cache
            WHERE route = ?
              AND normalized_query LIKE '% | %'
              AND {structured_conditions}
            """,
            [route] + structured_params,
        )
        deleted += cursor.rowcount

        # Phase 2: free-text entries — no ' | ' separator — that contain
        # any of the raw values as plain substrings.
        if include_freetext:
            freetext_conditions = " OR ".join(
                "normalized_query LIKE ?" for _ in filters
            )
            freetext_params = [
                f"%{value.strip().lower()}%"
                for value in filters.values()
            ]
            cursor = conn.execute(
                f"""
                DELETE FROM semantic_cache
                WHERE route = ?
                  AND normalized_query NOT LIKE '% | %'
                  AND ({freetext_conditions})
                """,
                [route] + freetext_params,
            )
            deleted += cursor.rowcount

        conn.commit()

    logger.info(
        "Cache invalidation complete. filters=%s route=%s deleted=%d",
        filters,
        route,
        deleted,
    )

    return deleted


# ── Convenience wrappers ─────────────────────────────────────────────────────

def invalidate_cache_by_destination(destination_city: str, **kwargs) -> int:
    """Invalidate all cache entries for a given destination city."""
    return invalidate_cache_by_fields(
        {"destination_city": destination_city.strip().lower()},
        **kwargs,
    )


def invalidate_cache_by_airline(airline: str, **kwargs) -> int:
    """Invalidate all cache entries that mention a specific airline."""
    return invalidate_cache_by_fields(
        {"airline": airline.strip().lower()},
        **kwargs,
    )


def invalidate_cache_by_budget_bucket(budget_bucket: str, **kwargs) -> int:
    """Invalidate all cache entries for a given budget bucket."""
    return invalidate_cache_by_fields(
        {"budget_bucket": str(budget_bucket).strip().lower()},
        **kwargs,
    )


# ── DB-change notification registry ─────────────────────────────────────────

# Maps each DB table name to the cache key fields it affects.
# A table can affect multiple fields (e.g. flights affects both destination
# and origin). Add new tables here as the project grows — no other code changes
# needed for invalidation to work automatically.
_DB_TABLE_CACHE_FIELD_MAP: Dict[str, List[str]] = {
    "flights":     ["destination_city", "origin_airport"],
    "hotels":      ["destination_city"],
    "activities":  ["destination_city"],
    "restaurants": ["destination_city"],
    "weather":     ["destination_city"],
    "visa":        ["destination_city", "origin_country"],
}


def notify_db_changed(
    table: str,
    values: Dict[str, str],
    *,
    route: str = "cache_check",
) -> int:
    """
    Call this whenever a DB table is updated to automatically invalidate
    all stale cache entries related to the change.

    table  — the DB table that changed (e.g. "hotels", "flights")
    values — the affected field values (e.g. {"destination_city": "paris"})

    The registry (_DB_TABLE_CACHE_FIELD_MAP) decides which cache fields to
    invalidate based on the table. Only fields present in both the registry
    AND the provided values dict are used as filters — unspecified fields
    are ignored, so partial updates are safe.

    Returns the number of deleted cache entries.

    Example:
        notify_db_changed("hotels", {"destination_city": "Paris"})
        notify_db_changed("flights", {"destination_city": "Tokyo", "origin_airport": "TLV"})
    """
    cache_fields = _DB_TABLE_CACHE_FIELD_MAP.get(table.lower())

    if not cache_fields:
        logger.warning(
            "notify_db_changed: table=%s not in registry — add it to "
            "_DB_TABLE_CACHE_FIELD_MAP to enable auto-invalidation.",
            table,
        )
        return 0

    # Normalise values and keep only fields the registry cares about.
    filters = {
        field: values[field].strip().lower()
        for field in cache_fields
        if field in values
    }

    if not filters:
        logger.warning(
            "notify_db_changed: table=%s — none of the registry fields %s "
            "were present in values=%s. No entries invalidated.",
            table,
            cache_fields,
            values,
        )
        return 0

    deleted = invalidate_cache_by_fields(filters, route=route)

    logger.info(
        "notify_db_changed: table=%s filters=%s deleted=%d",
        table,
        filters,
        deleted,
    )

    return deleted