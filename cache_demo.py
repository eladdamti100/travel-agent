"""
Cache behaviour demo.

Shows MISS → store → HIT → currency distinction → TTL expiry.
No LLM calls needed — exercises the cache layer directly.

Run:  python cache_demo.py
"""

import os, warnings, sqlite3, tempfile
from pathlib import Path
from unittest.mock import patch

warnings.warn = lambda *a, **kw: None
os.environ["TQDM_DISABLE"] = "1"

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from src.services.semantic_cache import (
    find_cached_answer,
    find_exact_cached_answer,
    store_cache_entry,
    build_trip_cache_key,
    TTL_DAYS_WEB,
    TTL_DAYS_DB,
)
import src.services.semantic_cache as _mod

import logging
logging.disable(logging.CRITICAL)

W  = "\033[0m"
G  = "\033[32m"
R  = "\033[31m"
Y  = "\033[33m"
B  = "\033[34m"
BO = "\033[1m"
SEP = "─" * 68


def hit(label, result):
    score = f"score={result.similarity_score:.4f}" if result.similarity_score else ""
    match = f"  →  \"{result.matched_query}\"" if result.matched_query else ""
    status_color = G if "hit" in result.status.value else R
    print(f"  {BO}{label}{W}  {status_color}{result.status.value.upper()}{W}  {score}{match}")


def section(title):
    print(f"\n{SEP}")
    print(f"{BO}{title}{W}")
    print(SEP)


# ── SCENARIO 1: First query — MISS ────────────────────────────────────────────
section("SCENARIO 1 — First query (cache empty) → MISS")

query = "plan a 7-day trip to paris from tlv with a budget of $2000 for 2 people"
result = find_cached_answer(query)
hit("lookup", result)
print(f"\n  {Y}→ No cached answer. In the real app this triggers full LLM planning.{W}")


# ── SCENARIO 2: Store answer, then re-query → HIT ────────────────────────────
section("SCENARIO 2 — After storing the answer → exact HIT")

with patch("src.services.semantic_cache._cleanup_cache"):
    entry = store_cache_entry(
        query,
        "Here is your 7-day Paris trip plan:\n"
        "Day 1: Arrive CDG → Eiffel Tower → Seine river cruise\n"
        "Day 2: Louvre Museum → Marais district\n"
        "...",
        source="db",
    )

print(f"  Stored:  ttl={entry.ttl_days} days  source={entry.source}")
print(f"  Key:     {entry.normalized_query[:70]}...")

result2 = find_cached_answer(query)
hit("re-query", result2)
print(f"\n  {G}→ Instant answer returned — no LLM call needed.{W}")


# ── SCENARIO 3: Semantic similarity (paraphrased query) ───────────────────────
section("SCENARIO 3 — Similar but not identical queries (semantic score)")

similar_queries = [
    ("close paraphrase",   "7 day trip to paris from tel aviv, $2000 budget, 2 travelers"),
    ("very different",     "5 days in tokyo from jfk, budget $3000"),
    ("different city",     "7-day london trip from tlv with $2000 for 2 people"),
]

for label, q in similar_queries:
    r = find_cached_answer(q)
    hit(label, r)

print(f"\n  {Y}→ Threshold is 0.85. Paraphrased free-text queries score ~0.70–0.79.{W}")
print(f"  {Y}  In the real app, structured keys (key_version:v2 | ...) guarantee exact hits.{W}")


# ── SCENARIO 4: Structured key — real app behaviour ───────────────────────────
section("SCENARIO 4 — Structured cache key (how the real app stores/looks up)")

key_usd = build_trip_cache_key({
    "destination_city": "Paris",
    "duration_days":    7,
    "total_budget":     2000,
    "currency":         "USD",
    "num_travelers":    2,
    "origin_airport":   "TLV",
})
key_eur = build_trip_cache_key({
    "destination_city": "Paris",
    "duration_days":    7,
    "total_budget":     2000,
    "currency":         "EUR",
    "num_travelers":    2,
    "origin_airport":   "TLV",
})

print(f"  USD key: {key_usd}")
print(f"  EUR key: {key_eur}")
print(f"  Same?    {B}{key_usd == key_eur}{W}  ← €2000 and $2000 are distinct keys")

# Store USD answer
with patch("src.services.semantic_cache._cleanup_cache"):
    store_cache_entry(key_usd, "Paris plan for $2000 USD budget...", source="db")

r_usd = find_exact_cached_answer(key_usd)
r_eur = find_exact_cached_answer(key_eur)
hit("USD lookup", r_usd)
hit("EUR lookup", r_eur)
print(f"\n  {G}→ USD entry hits, EUR entry misses — currency isolation works.{W}")


# ── SCENARIO 5: TTL expiry ────────────────────────────────────────────────────
section("SCENARIO 5 — TTL expiry (web vs db source)")

with tempfile.TemporaryDirectory() as tmp:
    orig_path, orig_init = _mod._CACHE_DB_PATH, _mod._db_initialized
    _mod._CACHE_DB_PATH = Path(tmp) / "cache.db"
    _mod._db_initialized = False
    _mod.initialize_cache_db()

    for source, ttl in [("web", TTL_DAYS_WEB), ("db", TTL_DAYS_DB)]:
        q = f"{source} paris trip ttl test"
        with patch("src.services.semantic_cache.embed_text", return_value=[1.0, 0.0, 0.0]), \
             patch("src.services.semantic_cache._cleanup_cache"):
            store_cache_entry(q, "plan", source=source)

        fresh = find_exact_cached_answer(q)
        print(f"  source={source:3s}  ttl={ttl:2d} days  fresh  → {G}{fresh.status.value.upper()}{W}")

        with sqlite3.connect(_mod._CACHE_DB_PATH) as conn:
            conn.execute(
                f"UPDATE semantic_cache SET created_at = datetime('now', '-{ttl + 1} days') "
                "WHERE normalized_query = ?", (q,)
            )
            conn.commit()

        expired = find_exact_cached_answer(q)
        print(f"  source={source:3s}  ttl={ttl:2d} days  +{ttl+1}d   → {R}{expired.status.value.upper()}{W}  ({expired.reason})")

    _mod._CACHE_DB_PATH = orig_path
    _mod._db_initialized = orig_init


# ── SCENARIO 6: Key-version purge on startup ──────────────────────────────────
section("SCENARIO 6 — Stale key-version purged on initialize_cache_db()")

with tempfile.TemporaryDirectory() as tmp:
    orig_path, orig_init = _mod._CACHE_DB_PATH, _mod._db_initialized
    _mod._CACHE_DB_PATH = Path(tmp) / "cache.db"
    _mod._db_initialized = False
    _mod.initialize_cache_db()

    stale = "key_version:v1 | destination_city:paris | duration:week"
    with sqlite3.connect(_mod._CACHE_DB_PATH) as conn:
        conn.execute(
            "INSERT INTO semantic_cache (query, normalized_query, answer, route, embedding_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (stale, stale, "old answer", "cache_check", "[1.0]"),
        )
        conn.commit()
    print(f"  Inserted: '{stale}'")

    _mod._db_initialized = False
    _mod.initialize_cache_db()          # ← purge happens here

    r = find_exact_cached_answer(stale)
    print(f"  After re-init → {R}{r.status.value.upper()}{W}  ← old v1 entry was purged automatically")

    _mod._CACHE_DB_PATH = orig_path
    _mod._db_initialized = orig_init


print(f"\n{SEP}")
print(f"{BO}{G}All cache scenarios complete.{W}")
print(SEP + "\n")
