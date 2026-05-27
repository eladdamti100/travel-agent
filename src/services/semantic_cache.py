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

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Suppress all noisy loggers
import logging as _logging
_logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
# Disable tqdm progress bars used by sentence_transformers
os.environ["TQDM_DISABLE"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.cache import CacheCheckResult, CacheEntry, CacheStatus
from src.utils.logger import get_logger

logger = get_logger("semantic_cache")

_CACHE_DB_PATH = Path(__file__).parent.parent.parent / "data" / "semantic_cache.db"
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HIT_THRESHOLD = 0.85
_MAX_ROWS_PER_ROUTE = 200
_MAX_AGE_DAYS = 30

_embedding_model: Optional[SentenceTransformer] = None
_db_initialized: bool = False


def _get_embedding_model() -> SentenceTransformer:
    """
    Lazily loads and returns the local sentence-transformers embedding model.

    The model is cached in memory after the first load.
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)

    return _embedding_model


def initialize_cache_db() -> None:
    """
    Creates the semantic cache SQLite database and table if they do not exist.

    Guarded by _db_initialized so schema checks run only once per process.
    """
    global _db_initialized
    if _db_initialized:
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
                created_at TEXT NOT NULL
            )
            """
        )

        # Migrate existing databases that predate these columns.
        for migration in (
            "ALTER TABLE semantic_cache ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
            "ALTER TABLE semantic_cache ADD COLUMN compressed_answer TEXT NOT NULL DEFAULT ''",
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

        conn.commit()

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
            SELECT id, query, embedding_json
            FROM semantic_cache
            WHERE route = ?
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
            SELECT query, answer, compressed_answer
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

    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT query, answer, compressed_answer
            FROM semantic_cache
            WHERE route = ? AND normalized_query = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (route, normalized_query),
        ).fetchone()

    if not row:
        logger.info(
            "Exact cache MISS. route=%s normalized_query=%s",
            route,
            normalized_query,
        )

        return CacheCheckResult(
            status=CacheStatus.MISS,
            similarity_score=0.0,
            matched_query=None,
            cached_answer=None,
            cached_compressed_answer=None,
            reason="No exact structured cache match was found.",
        )

    logger.info(
        "Exact cache HIT. route=%s matched_query=%s",
        route,
        row["query"],
    )

    return CacheCheckResult(
        status=CacheStatus.HIT,
        similarity_score=1.0,
        matched_query=row["query"],
        cached_answer=row["answer"],
        cached_compressed_answer=row["compressed_answer"] or None,
        reason="Found an exact structured cache match.",
    )

def find_cached_answer(
    query: str,
    *,
    route: str = "cache_check",
    threshold: float = DEFAULT_HIT_THRESHOLD,
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

    exact_result = find_exact_cached_answer(query=query, route=route)
    if exact_result.status == CacheStatus.HIT:
        return exact_result

    normalized_query = normalize_query(query)

    logger.info(
        "Storing cache entry. route=%s normalized_query=%s",
        route,
        normalized_query,
    )

    query_embedding = embed_text(normalized_query)

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

    for row in index_rows:
        try:
            cached_embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, cached_embedding)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            logger.warning("Skipping invalid cache embedding: %s", error)
            continue

        if score > best_score:
            best_score = score
            best_id = row["id"]
            best_query = row["query"]

    if best_id is not None and best_score >= threshold:
        full_row = _fetch_row_by_id(best_id)

        logger.info(
            "Semantic cache hit. score=%.4f matched_query=%s",
            best_score,
            best_query,
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
        )

    logger.info("Semantic cache miss. best_score=%.4f threshold=%.4f", best_score, threshold)

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
) -> CacheEntry:
    """
    Stores a new semantic cache entry.

    This should be called only after a successful final answer was produced
    for a full trip-planning request.
    """
    initialize_cache_db()

    normalized_query = normalize_query(query)
    embedding = embed_text(normalized_query)
    now = datetime.now(timezone.utc)

    entry = CacheEntry(
        query=query,
        normalized_query=normalized_query,
        answer=answer,
        route=route,
        embedding=embedding,
        confidence=confidence,
        timestamp=now,
        compressed_answer=compressed_answer,
    )

    with sqlite3.connect(_CACHE_DB_PATH) as conn:
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
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.query,
                entry.normalized_query,
                entry.answer,
                entry.route,
                json.dumps(entry.embedding),
                entry.confidence,
                entry.compressed_answer,
                entry.timestamp.isoformat(),
            ),
        )

        conn.commit()

        logger.info(
        "Cache entry committed. route=%s query=%s",
        route,
        query,
        )

    _cleanup_cache(route=route)

    logger.info("Stored semantic cache entry for route=%s query=%s confidence=%.4f", route, query, confidence)

    return entry

def normalize_budget_bucket(total_budget: float | None) -> str:
    """
    Normalizes budget into stable buckets to avoid cache misses caused by tiny differences.
    """
    if total_budget is None:
        return "unknown"

    try:
        budget = float(total_budget)
    except (TypeError, ValueError):
        return "unknown"

    if budget <= 0:
        return "unknown"

    rounded = round(budget / 100) * 100
    return str(int(rounded))


def build_trip_cache_key(
    *,
    origin_airport: str | None,
    origin_country: str | None,
    destination_city: str | None,
    duration_days: int | None,
    total_budget: float | None,
) -> str | None:
    """
    Builds a deterministic structured cache key for full trip-planning requests.

    Returns None when the key does not have enough required planning fields.
    """
    if not origin_airport or not origin_country or not destination_city or not duration_days:
        return None

    budget_key = normalize_budget_bucket(total_budget)

    return normalize_query(
        " | ".join(
            [
                f"origin_airport:{origin_airport.strip().upper()}",
                f"origin_country:{origin_country.strip().title()}",
                f"destination_city:{destination_city.strip().title()}",
                f"duration_days:{int(duration_days)}",
                f"budget_bucket:{budget_key}",
            ]
        )
    )

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
              AND created_at < datetime('now', ? || ' days')
            """,
            (route, f"-{_MAX_AGE_DAYS}"),
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

    logger.debug("Cache cleanup done for route=%s (max_rows=%d, max_age_days=%d)", route, _MAX_ROWS_PER_ROUTE, _MAX_AGE_DAYS)