"""
LangGraph workflow — compiles the full travel planner state graph.

Graph topology
──────────────
START → extract_metadata → validator
          │ blocked        → END
          │ HITL resume    → resume_hitl_context → master_planner
          └─ approved      → master_orchestrator
                               ├─ preferences_memory → summarizer → END
                               ├─ researcher → END
                               └─ cache_check
                                    ├─ hit  → END
                                    └─ miss → master_planner
                                                  └─ critic
                                                       ├─ rejected → master_planner (replan)
                                                       └─ passed   → hitl_approval
                                                                         ├─ approved  → cache_store → summarizer → END
                                                                         ├─ edit      → master_planner
                                                                         └─ cancelled → END

Checkpointing:
  CLI path  (synchronous) — SqliteSaver via a persistent sqlite3 connection.
  API path  (async)       — AsyncSqliteSaver, bootstrapped during FastAPI startup.
  Both write to the same database file so sessions are shared across interfaces.

Invariant:
  Every user query passes through validator before intent routing.
"""

import logging
import sqlite3

logging.getLogger("langgraph").setLevel(logging.ERROR)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.config.settings import settings
from src.graph.nodes import (
    cache_check_node,
    cache_store_node,
    critic_node,
    extract_metadata,
    hitl_approval_node,
    master_orchestrator_node,
    master_planner_node,
    preferences_memory_node,
    researcher_node,
    resume_hitl_context_node,
    run_validator,
    summarizer_node,
)
from src.graph.router import (
    route_after_cache_check,
    route_after_critic,
    route_after_hitl,
    route_after_master_planner,
    route_after_orchestrator,
    route_after_validator,
)
from src.graph.state import AgentState

settings.cache_dir.mkdir(parents=True, exist_ok=True)

# Synchronous SQLite connection — used by the CLI and synchronous graph.stream().
_conn = sqlite3.connect(str(settings.checkpoints_db_path), check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def build_graph(checkpointer=None):
    """
    Compile and return the StateGraph.

    checkpointer defaults to the module-level SqliteSaver (synchronous CLI path).
    Pass an AsyncSqliteSaver instance for the async FastAPI path.
    """
    cp = checkpointer if checkpointer is not None else _checkpointer
    builder = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("extract_metadata",    extract_metadata)
    builder.add_node("validator",           run_validator)
    builder.add_node("resume_hitl_context", resume_hitl_context_node)
    builder.add_node("master_orchestrator", master_orchestrator_node)
    builder.add_node("preferences_memory",  preferences_memory_node)
    builder.add_node("researcher",          researcher_node)
    builder.add_node("cache_check",         cache_check_node)
    builder.add_node("master_planner",      master_planner_node)
    builder.add_node("critic",              critic_node)
    builder.add_node("hitl_approval",       hitl_approval_node)
    builder.add_node("cache_store",         cache_store_node)
    builder.add_node("summarizer",          summarizer_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_edge(START, "extract_metadata")
    builder.add_edge("extract_metadata", "validator")

    builder.add_conditional_edges(
        "validator",
        route_after_validator,
        {
            "resume_hitl_context": "resume_hitl_context",
            "master_orchestrator": "master_orchestrator",
            END: END,
        },
    )

    builder.add_edge("resume_hitl_context", "master_planner")

    builder.add_conditional_edges(
        "master_orchestrator",
        route_after_orchestrator,
        {
            "preferences_memory": "preferences_memory",
            "researcher":         "researcher",
            "cache_check":        "cache_check",
        },
    )

    builder.add_edge("preferences_memory", "summarizer")
    builder.add_edge("researcher", END)

    builder.add_conditional_edges(
        "cache_check",
        route_after_cache_check,
        {"master_planner": "master_planner", END: END},
    )

    builder.add_conditional_edges(
        "master_planner",
        route_after_master_planner,
        {"critic": "critic", END: END},
    )

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"master_planner": "master_planner", "hitl_approval": "hitl_approval"},
    )

    builder.add_conditional_edges(
        "hitl_approval",
        route_after_hitl,
        {"cache_store": "cache_store", "master_planner": "master_planner", END: END},
    )

    builder.add_edge("cache_store", "summarizer")
    builder.add_edge("summarizer", END)

    return builder.compile(checkpointer=cp)


# Module-level singleton used by main.py (synchronous CLI path).
graph = build_graph()
