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
    0.86
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

import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.cache import CacheCheckResult, CacheEntry, CacheStatus
from src.utils.logger import get_logger

logger = get_logger("semantic_cache")

_CACHE_DB_PATH = Path(__file__).parent.parent.parent / "data" / "semantic_cache.db"
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HIT_THRESHOLD = 0.85

_embedding_model: Optional[SentenceTransformer] = None
_db_initialized: bool = False


def _get_embedding_model() -> SentenceTransformer:
    """
    Lazily loads and returns the local sentence-transformers embedding model.

    The model is cached in memory after the first load.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading embedding model: %s", _EMBEDDING_MODEL_NAME)
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


def _load_cache_rows(route: str = "cache_check") -> List[Dict[str, Any]]:
    """
    Loads cache rows for a specific route.
    """
    initialize_cache_db()

    with sqlite3.connect(_CACHE_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT query, normalized_query, answer, route, embedding_json, compressed_answer, created_at
            FROM semantic_cache
            WHERE route = ?
            ORDER BY id DESC
            """,
            (route,),
        ).fetchall()

    return [dict(row) for row in rows]


def find_cached_answer(
    query: str,
    *,
    route: str = "cache_check",
    threshold: float = DEFAULT_HIT_THRESHOLD,
) -> CacheCheckResult:
    """
    Finds the best semantic cache match for the given query.

    Returns HIT if the best cosine similarity is at least threshold.
    Otherwise returns MISS.
    """
    initialize_cache_db()

    normalized_query = normalize_query(query)
    query_embedding = embed_text(normalized_query)

    rows = _load_cache_rows(route=route)

    if not rows:
        return CacheCheckResult(
            status=CacheStatus.MISS,
            similarity_score=0.0,
            matched_query=None,
            cached_answer=None,
            reason="Semantic cache is empty for this route.",
        )

    best_score = 0.0
    best_row: Optional[Dict[str, Any]] = None

    for row in rows:
        try:
            cached_embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query_embedding, cached_embedding)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            logger.warning("Skipping invalid cache embedding: %s", error)
            continue

        if score > best_score:
            best_score = score
            best_row = row

    if best_row and best_score >= threshold:
        logger.info(
            "Semantic cache hit. score=%.4f matched_query=%s",
            best_score,
            best_row["query"],
        )

        return CacheCheckResult(
            status=CacheStatus.HIT,
            similarity_score=best_score,
            matched_query=best_row["query"],
            cached_answer=best_row["answer"],
            cached_compressed_answer=best_row.get("compressed_answer") or None,
            reason="Found a sufficiently similar cached answer.",
        )

    logger.info("Semantic cache miss. best_score=%.4f threshold=%.4f", best_score, threshold)

    return CacheCheckResult(
        status=CacheStatus.MISS,
        similarity_score=best_score,
        matched_query=best_row["query"] if best_row else None,
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

    _cleanup_cache(route=route)

    logger.info("Stored semantic cache entry for route=%s query=%s confidence=%.4f", route, query, confidence)

    return entry


_MAX_ROWS_PER_ROUTE = 200
_MAX_AGE_DAYS = 30


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