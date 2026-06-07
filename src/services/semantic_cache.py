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

import atexit
import functools
import json
import os
import random
import re
import sqlite3
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
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

# Background executor for cache cleanup/deletion — never blocks the caller.
_bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache_bg_cleanup")
atexit.register(_bg_executor.shutdown, wait=False)


def _bg_submit(fn: Callable, *args: Any) -> None:
    """Fire-and-forget: submit fn(*args) to the background cleanup thread."""
    try:
        future = _bg_executor.submit(fn, *args)
        future.add_done_callback(
            lambda f: logger.error("Background cache task raised: %s", f.exception())
            if f.exception() else None
        )
    except Exception as exc:
        logger.error("Failed to submit background cache task: %s", exc)


# ── Background sweep thread ───────────────────────────────────────────────────
# Runs every 6 hours regardless of query traffic. Removes entries that are past
# their valid_until date or TTL so the DB stays compact even when idle.

_SWEEP_INTERVAL_SECONDS = 6 * 60 * 60  # 6 hours
_sweep_stop = threading.Event()


def _sweep_expired_entries() -> int:
    """
    Deletes every cache row that has expired — either past valid_until
    (trip date has passed) or past its TTL (data is stale).

    Safe to call at any time: a missing DB is silently ignored.
    """
    if not _CACHE_DB_PATH.exists():
        return 0
    try:
        with sqlite3.connect(_CACHE_DB_PATH) as conn:
            deleted = conn.execute(
                """
                DELETE FROM semantic_cache
                WHERE (valid_until IS NOT NULL AND valid_until < date('now'))
                   OR created_at < datetime('now', '-' || ttl_days || ' days')
                """
            ).rowcount
        if deleted:
            logger.info("Periodic sweep: deleted %d expired cache entries.", deleted)
        return deleted
    except Exception as exc:
        logger.error("Periodic sweep failed: %s", exc)
        return 0


def _sweep_loop() -> None:
    """Daemon loop: sleep _SWEEP_INTERVAL_SECONDS, then sweep, repeat."""
    while not _sweep_stop.wait(_SWEEP_INTERVAL_SECONDS):
        _sweep_expired_entries()


_sweep_thread = threading.Thread(target=_sweep_loop, name="cache_sweep", daemon=True)
_sweep_thread.start()


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
            # WAL lets background writers and the main-thread reader run
            # concurrently without blocking each other.
            # synchronous=NORMAL is safe with WAL and avoids an fsync per write.
            # cache_size=-4000 keeps 4 MB of pages in RAM to reduce I/O.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-4000")

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
                    valid_until TEXT,
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
                "ALTER TABLE semantic_cache ADD COLUMN valid_until TEXT",
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

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_destination
                ON semantic_cache(LOWER(json_extract(trip_context_json, '$.destination_city')))
                """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_cache_valid_until
                ON semantic_cache(valid_until)
                WHERE valid_until IS NOT NULL
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


@functools.lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple:
    """LRU-cached embedding — same text always produces the same vector."""
    return tuple(_get_embedding_model().encode(text, normalize_embeddings=True).tolist())


def embed_text(text: str) -> List[float]:
    """
    Creates an embedding vector for the given text.
    Results are LRU-cached (maxsize=256) so repeated queries skip the neural net.
    """
    return list(_embed_cached(text))


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


def _load_embedding_index(
    route: str,
    trip_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Loads id, query, and embedding_json for candidate rows in a route.

    When trip_context is provided, two WHERE clauses are pushed into SQLite
    before any Python runs (pre-filtering):
      - destination_city exact match
      - total_budget within ±5%

    Rows without trip_context_json always pass through so free-text entries
    remain reachable via semantic fallback.
    """
    initialize_cache_db()

    conditions = [
        "route = ?",
        "created_at >= datetime('now', '-' || ttl_days || ' days')",
        "(valid_until IS NULL OR valid_until >= date('now'))",
    ]
    params: List[Any] = [route]

    if trip_context:
        destination = trip_context.get("destination_city")
        if destination:
            conditions.append(
                "(trip_context_json IS NULL"
                " OR LOWER(json_extract(trip_context_json, '$.destination_city')) = ?)"
            )
            params.append(str(destination).strip().lower())

        budget = trip_context.get("total_budget")
        if budget is not None:
            try:
                budget_f = float(budget)
                min_b, max_b = budget_f * 0.95, budget_f * 1.05
                conditions.append(
                    "(trip_context_json IS NULL"
                    " OR json_extract(trip_context_json, '$.total_budget') IS NULL"
                    " OR CAST(json_extract(trip_context_json, '$.total_budget') AS REAL)"
                    "    BETWEEN ? AND ?)"
                )
                params.extend([min_b, max_b])
            except (TypeError, ValueError):
                pass

    where = " AND ".join(conditions)

    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        # Include answer columns only when pre-filter is active (few rows expected).
        # Base conditions = 3 (route + TTL + valid_until); any extra means a
        # destination/budget filter was added — safe to load answer text.
        answer_cols = ", answer, compressed_answer, source, ttl_days" if len(conditions) > 3 else ""
        rows = conn.execute(
            f"""
            SELECT id, query, embedding_json, trip_context_json{answer_cols}
            FROM semantic_cache
            WHERE {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params + [_MAX_ROWS_PER_ROUTE],
        ).fetchall()

    logger.info(
        "Pre-filter loaded %d candidate rows. route=%s destination=%s budget=%s",
        len(rows),
        route,
        trip_context.get("destination_city") if trip_context else None,
        trip_context.get("total_budget") if trip_context else None,
    )

    return [dict(row) for row in rows]


def _compute_valid_until(trip_context: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Returns the ISO date (YYYY-MM-DD) after which this cached answer is no longer valid.

    Logic (in priority order):
      1. travel_start_date known → valid until start_date - 1 day
         (day before departure all time-sensitive items — flights, hotels, events — become irrelevant)
      2. travel_end_date known  → valid until end_date
         (plan is valid through the last day of the trip)
      3. Neither known → None  (rely on TTL only)
    """
    if not trip_context:
        return None
    start = trip_context.get("travel_start_date")
    if start:
        try:
            return (date.fromisoformat(start) - timedelta(days=1)).isoformat()
        except ValueError:
            pass
    end = trip_context.get("travel_end_date")
    if end:
        try:
            return date.fromisoformat(end).isoformat()
        except ValueError:
            pass
    return None


def _delete_by_id(row_id: int) -> None:
    """Deletes a single cache row by primary key. Safe for background use."""
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.execute("DELETE FROM semantic_cache WHERE id = ?", (row_id,))


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
    _normalized: Optional[str] = None,
) -> CacheCheckResult:
    """
    Finds an exact cache match by normalized_query before semantic similarity is used.

    Pass _normalized when the caller already holds normalize_query(query) to
    avoid computing it twice (find_cached_answer does this internally).
    """
    initialize_cache_db()

    normalized_query = _normalized if _normalized is not None else normalize_query(query)

    logger.info(
        "Exact cache lookup. route=%s normalized_query=%s",
        route,
        normalized_query,
    )

    # Fetch the latest row; validity is checked in Python so we can log the
    # exact reason. The WHERE already excludes rows past valid_until so only
    # truly live entries reach the TTL check.
    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, query, answer, compressed_answer, source, ttl_days,
                   CASE WHEN created_at >= datetime('now', '-' || ttl_days || ' days')
                        THEN 1 ELSE 0 END AS is_valid
            FROM semantic_cache
            WHERE route = ? AND normalized_query = ?
              AND (valid_until IS NULL OR valid_until >= date('now'))
            ORDER BY id DESC
            LIMIT 1
            """,
            (route, normalized_query),
        ).fetchone()

    if not row:
        miss_reason = "No cache entry found for this query."
    elif not row["is_valid"]:
        miss_reason = "Cache entry expired (TTL exceeded)."
        # Delete by id — not by normalized_query — to avoid erasing a
        # concurrently stored fresh entry with the same text.
        _bg_submit(_delete_by_id, row["id"])
        logger.info(
            "Scheduled async delete of expired entry id=%s route=%s",
            row["id"],
            route,
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

    # Normalize once here; pass it to find_exact_cached_answer so the regex
    # pipeline runs exactly once per find_cached_answer call.
    normalized_query = normalize_query(query)

    # Exact match is the fast path — run it before any cleanup overhead.
    exact_result = find_exact_cached_answer(query=query, route=route, _normalized=normalized_query)
    if exact_result.status == CacheStatus.HIT:
        return exact_result

    # Housekeeping: TTL and valid_until correctness are enforced by WHERE clauses,
    # so cleanup is purely space management and can be both probabilistic and async.
    if random.random() < _CLEANUP_ON_LOOKUP_PROBABILITY:
        _bg_submit(_purge_expired_web_entries, route)
        _bg_submit(_cleanup_cache, route)

    logger.info(
        "Semantic cache lookup. route=%s normalized_query=%s",
        route,
        normalized_query,
    )

    query_embedding = embed_text(normalized_query)

    # Compute coverage for threshold adjustment.
    # Destination + budget are enforced by SQLite pre-filtering; only duration
    # still needs a Python hard filter inside the scan loop below.
    effective_threshold = threshold
    _check_hard_filters = None
    if trip_context:
        try:
            from src.services.trip_vector import (
                adjusted_threshold,
                check_hard_filters as _check_hard_filters,
                compute_coverage,
            )
            coverage = compute_coverage(trip_context)
            effective_threshold = adjusted_threshold(threshold, coverage)
            logger.info(
                "Coverage=%.2f threshold %.2f→%.2f",
                coverage, threshold, effective_threshold,
            )
        except Exception as exc:
            logger.warning("Coverage computation failed — using base threshold: %s", exc)

    index_rows = _load_embedding_index(route=route, trip_context=trip_context)

    if not index_rows:
        return CacheCheckResult(
            status=CacheStatus.MISS,
            similarity_score=0.0,
            matched_query=None,
            cached_answer=None,
            reason="Semantic cache is empty for this route.",
        )

    # ── Unified cosine scan ───────────────────────────────────────────────────
    # Destination + budget are pre-filtered by SQLite.
    # Duration hard filter runs here in Python for rows that have trip_context_json.
    candidate_rows: List[Dict[str, Any]] = []
    candidate_vecs: List[List[float]] = []

    for row in index_rows:
        if _check_hard_filters is not None and row.get("trip_context_json"):
            try:
                cached_ctx = json.loads(row["trip_context_json"])
                passes, reason = _check_hard_filters(trip_context, cached_ctx)
                if not passes:
                    logger.debug("Duration filter rejected row id=%s: %s", row["id"], reason)
                    continue
            except (json.JSONDecodeError, TypeError, ValueError) as err:
                logger.warning("Skipping row with invalid trip_context_json id=%s: %s", row["id"], err)
                continue
        try:
            candidate_vecs.append(json.loads(row["embedding_json"]))
            candidate_rows.append(row)
        except (json.JSONDecodeError, TypeError, ValueError) as err:
            logger.warning("Skipping row with invalid embedding id=%s: %s", row["id"], err)

    best_score = 0.0
    best_row: Optional[Dict[str, Any]] = None

    if candidate_vecs:
        q_vec = np.array(query_embedding, dtype=np.float32)
        matrix = np.array(candidate_vecs, dtype=np.float32)
        scores = matrix @ q_vec
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_row = candidate_rows[best_idx]

    # Clamp to [0, 1] — dot products on unit vectors can produce 1.0000001.
    best_score = max(0.0, min(best_score, 1.0))

    if best_row is not None and best_score >= effective_threshold:
        # When no pre-filter was active, _load_embedding_index omitted the answer
        # columns to avoid loading large text for every scanned row. Fetch them
        # now for this single winning row.
        if best_row.get("answer") is None:
            full = _fetch_row_by_id(best_row["id"])
            if full:
                best_row = {**best_row, **full}

        logger.info(
            "Semantic cache hit. score=%.4f matched_query=%s source=%s ttl_days=%s",
            best_score,
            best_row.get("query"),
            best_row.get("source"),
            best_row.get("ttl_days"),
        )

        return CacheCheckResult(
            status=CacheStatus.HIT,
            similarity_score=best_score,
            matched_query=best_row.get("query"),
            cached_answer=best_row.get("answer"),
            cached_compressed_answer=best_row.get("compressed_answer") or None,
            reason="Found a sufficiently similar cached answer.",
            source=best_row.get("source"),
            ttl_days=best_row.get("ttl_days"),
        )

    logger.info("Semantic cache miss. best_score=%.4f effective_threshold=%.4f", best_score, effective_threshold)

    return CacheCheckResult(
        status=CacheStatus.MISS,
        similarity_score=best_score,
        matched_query=best_row.get("query") if best_row else None,
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

    # Compute valid_until from trip dates so the entry auto-expires when the
    # travel window passes — regardless of when it was stored.
    valid_until = _compute_valid_until(trip_context)
    if valid_until:
        logger.info("Cache entry valid_until=%s query=%s", valid_until, query)

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
                valid_until,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                valid_until,
                entry.timestamp.isoformat(),
            ),
        )

        conn.commit()

    logger.info("Cache entry committed. route=%s query=%s", route, query)

    # Cleanup runs async — never blocks the caller.
    _bg_submit(_cleanup_cache, route)

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