"""
LangGraph workflow — compiles the full travel planner state graph.

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
  ├─ [blocked]            → END
  ├─ [resume_hitl_context]→ resume_hitl_context → master_planner
  └─ [approved]           → master_orchestrator
                                │
                                ▼ route_after_orchestrator
                                ├─ [preferences_memory] → preferences_memory → summarizer → END
                                ├─ [research]           → researcher → END
                                └─ [cache_check]        → cache_check
                                                               │
                                                               ▼ route_after_cache_check
                                                               ├─ [hit]  → END
                                                               └─ [miss] → master_planner
                                                                               │
                                                                               ▼ route_after_master_planner
                                                                               ├─ [missing_required_info] → END
                                                                               └─ [final_plan] → critic
                                                                                                    │
                                                                                                    ▼ route_after_critic
                                                                                                    ├─ [rejected] → master_planner (replan, attempt++)
                                                                                                    └─ [approved] → hitl_approval
                                                                                                                        │
                                                                                                                        ▼ route_after_hitl
                                                                                                                        ├─ [approved]  → cache_store → summarizer → END
                                                                                                                        ├─ [edit]      → master_planner (with hitl_feedback)
                                                                                                                        └─ [cancelled] → END

HITL Resume Flow:
  When a previous planner turn stopped to ask for missing trip details, the next
  user reply bypasses master_orchestrator, researcher, and semantic cache, and
  resumes planning directly from the saved pending TripContext.

Plan-approval HITL:
  After a complete plan is produced, the critic runs and then the graph suspends
  for user approval. The user can approve (proceed to cache), edit (replan with
  feedback), or cancel (end gracefully).

Legacy path:
  The agent/tools loop remains registered for backward compatibility.
  cache_miss now routes to master_planner.

Strict rule:
  Every user query must pass through validator before intent routing.
"""

import logging
import sqlite3
from pathlib import Path

logging.getLogger("langgraph").setLevel(logging.ERROR)

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    build_tools_node,
    cache_check_node,
    cache_store_node,
    call_model,
    circuit_breaker,
    critic_node,
    extract_metadata,
    hitl_approval_node,
    master_orchestrator_node,
    master_planner_node,
    preferences_memory_node,
    researcher_node,
    resume_hitl_context_node,
    reviewer_node,
    run_validator,
    summarizer_node,
)
from src.graph.router import (
    route_after_cache_check,
    route_after_critic,
    route_after_hitl,
    route_after_master_planner,
    route_after_metadata,
    route_after_orchestrator,
    route_after_validator,
    should_continue,
)
from src.graph.state import AgentState

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
    builder.add_node("resume_hitl_context", resume_hitl_context_node)
    builder.add_node("master_orchestrator", master_orchestrator_node)
    builder.add_node("preferences_memory", preferences_memory_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("cache_check", cache_check_node)
    builder.add_node("master_planner", master_planner_node)
    builder.add_node("critic", critic_node)
    builder.add_node("hitl_approval", hitl_approval_node)
    builder.add_node("cache_store", cache_store_node)

    # Legacy nodes kept for the agent/tools loop path.
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
        {"validator": "validator"},
    )

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
            "researcher": "researcher",
            "cache_check": "cache_check",
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
        {
            "critic": "critic",
            END: END,
        },
    )

    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "master_planner": "master_planner",
            "hitl_approval": "hitl_approval",
        },
    )

    builder.add_conditional_edges(
        "hitl_approval",
        route_after_hitl,
        {
            "cache_store": "cache_store",
            "master_planner": "master_planner",
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
