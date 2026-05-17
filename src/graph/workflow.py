import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    build_tools_node,
    cache_check_node,
    cache_store_node,
    call_model,
    circuit_breaker,
    extract_metadata,
    master_orchestrator_node,
    preferences_memory_node,
    researcher_node,
    reviewer_node,
    run_validator,
    summarizer_node,
)
from src.graph.router import (
    route_after_cache_check,
    route_after_metadata,
    route_after_orchestrator,
    route_after_validator,
    should_continue,
)
from src.graph.state import AgentState

"""
Graph topology
──────────────
START
  │
  ▼
extract_metadata
  │
  ▼
validator
  │
  ▼ route_after_validator
  ├─ [blocked]  → END
  └─ [approved] → master_orchestrator
                      │
                      ▼ route_after_orchestrator
                      ├─ [preferences_memory] → preferences_memory_node → summarizer → END
                      ├─ [research]           → researcher_node ───────→ END
                      └─ [cache_check]        → cache_check
                                                     │
                                                     ▼ route_after_cache_check
                                                     ├─ [cache_hit]  → END
                                                     └─ [cache_miss] → agent ◄──────────┐
                                                                         │              │
                                                                         ├─ tools ──────┘
                                                                         ├─ circuit_breaker → END
                                                                         ├─ reviewer → summarizer → END
                                                                         └─ cache_store → summarizer → END

Strict rule:
  Every user query must pass through validator before intent routing.

Current phase:
  preferences_memory and research are implemented as dedicated agent routes.
  cache_check is implemented with semantic embeddings.
  cache_store saves successful cache-miss answers for future cache hits.

Next phase:
  cache_miss will route to the dedicated planner instead of the legacy agent.
"""

# SQLite connection created once at module level — stays open for app lifetime.
_DB_PATH = Path(__file__).parent.parent.parent / "data" / "checkpoints.db"
_conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def build_graph():
    """
    Compile and return the StateGraph with SqliteSaver for persistent
    cross-session memory.

    Each unique thread_id is a separate conversation.
    Admin sessions route through the reviewer node after full plans.
    All normal final answers route through the summarizer for compact memory.
    """
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("extract_metadata", extract_metadata)
    builder.add_node("validator", run_validator)
    builder.add_node("master_orchestrator", master_orchestrator_node)
    builder.add_node("preferences_memory", preferences_memory_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("cache_check", cache_check_node)
    builder.add_node("cache_store", cache_store_node)
    builder.add_node("agent", call_model)
    builder.add_node("tools", build_tools_node())
    builder.add_node("circuit_breaker", circuit_breaker)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("summarizer", summarizer_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "extract_metadata")

    builder.add_conditional_edges(
        "extract_metadata",
        route_after_metadata,
        {
            "validator": "validator",
        },
    )

    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "master_orchestrator": "master_orchestrator",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "master_orchestrator",
        route_after_orchestrator,
        {
            "preferences_memory": "preferences_memory",
            "researcher": "researcher",
            "cache_check": "cache_check",
        },
    )

    builder.add_edge("preferences_memory", "summarizer")
    builder.add_edge("researcher", END)

    builder.add_conditional_edges(
        "cache_check",
        route_after_cache_check,
        {
            "agent": "agent",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "circuit_breaker": "circuit_breaker",
            "reviewer": "reviewer",
            "cache_store": "cache_store",
            "summarizer": "summarizer",
        },
    )

    builder.add_edge("tools", "agent")
    builder.add_edge("circuit_breaker", END)
    builder.add_edge("reviewer", "summarizer")
    builder.add_edge("cache_store", "summarizer")
    builder.add_edge("summarizer", END)

    return builder.compile(checkpointer=_checkpointer)


# Module-level singleton used by main.py.
graph = build_graph()