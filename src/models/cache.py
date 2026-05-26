from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class CacheStatus(str, Enum):
    """
    Semantic cache lookup status.
    """
    HIT = "hit"
    MISS = "miss"


class CacheCheckResult(BaseModel):
    """
    Result returned by the semantic cache checker.
    """

    status: CacheStatus = Field(
        description="Whether the semantic cache found a sufficiently similar previous answer."
    )

    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Cosine similarity score between the current query and the best cached query."
    )

    matched_query: str | None = Field(
        default=None,
        description="The cached query that best matched the current query."
    )

    cached_answer: str | None = Field(
        default=None,
        description="The cached answer to return when this is a cache hit."
    )

    reason: str = Field(
        description="Short explanation of why the lookup was a cache hit or cache miss."
    )


class CacheEntry(BaseModel):
    """
    A stored semantic cache entry.
    """

    query: str = Field(
        description="Original user query."
    )

    normalized_query: str = Field(
        description="Normalized query used for comparison and search."
    )

    answer: str = Field(
        description="Final answer stored for reuse."
    )

    route: str = Field(
        description="Route that generated this answer, usually cache_check/planner."
    )

    embedding: list[float] = Field(
        description="Vector embedding of the normalized query."
    )

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score at store time (1.0 for freshly computed entries).",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this entry was stored.",
    )

    compressed_answer: str = Field(
        default="",
        description="LLM-generated bullet-point summary of the full answer.",
    )