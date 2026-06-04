"""
FastAPI HTTP server for Marco — AI Travel Planner.

Provides an HTTP alternative to the CLI REPL so the planner can be
integrated with web frontends, mobile apps, or external services.

Endpoints
─────────
POST /plan
    Send a message; receive the planner's response as JSON.
    Maintains session state via thread_id (same as the CLI).

POST /plan/{thread_id}/approve
    Approve, edit, or cancel a plan that is waiting at hitl_approval.

GET  /health
    Liveness check — returns {"status": "ok"}.

Run:  uvicorn src.api.app:app --reload
      python -m src.api.server
"""

from __future__ import annotations

import os

# HuggingFace / torch silence vars must precede any heavy import.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.config.settings import ConfigurationError, settings
from src.graph.workflow import build_graph
from src.models.final_plan import FinalPlan

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Marco — AI Travel Planner",
    description="Natural-language trip planning via a multi-agent LangGraph pipeline.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level graph reference — replaced with an async-checkpointer graph on startup.
_graph = None


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    global _graph
    try:
        settings.validate_startup()
    except ConfigurationError as exc:
        raise RuntimeError(f"Marco configuration error: {exc}") from exc

    settings.configure_tracing()

    # Build the graph with AsyncSqliteSaver for non-blocking checkpoint writes.
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoints_db_path)
    ) as async_cp:
        _graph = build_graph(checkpointer=async_cp)


def _get_graph():
    """Return the async-checkpointed graph, or fall back to the sync graph."""
    if _graph is not None:
        return _graph
    from src.graph.workflow import graph
    return graph


# ── Request / response models ─────────────────────────────────────────────────

class PlanRequest(BaseModel):
    message: str = Field(..., description="Natural-language trip request or reply.")
    thread_id: str = Field(
        default="default",
        description="Session identifier. Use the same value across turns to maintain conversation state.",
    )


class ApprovalRequest(BaseModel):
    decision: str = Field(..., description="'approved', 'edit', or 'cancelled'")
    feedback: str = Field(
        default="",
        description="Free-text feedback required when decision='edit'.",
    )


class PlanResponse(BaseModel):
    thread_id: str
    message: str
    awaiting_clarification: bool = False
    awaiting_approval: bool = False
    plan: Optional[Dict[str, Any]] = None   # FinalPlan.model_dump() when ready
    planner_status: Optional[str] = None


# ── Helper ────────────────────────────────────────────────────────────────────

def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _extract_last_ai_message(state: dict) -> str:
    """Return the last AIMessage content from state."""
    from langchain_core.messages import AIMessage
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
    return ""


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "provider": settings.llm_provider}


@app.post("/plan", response_model=PlanResponse)
async def plan(request: PlanRequest) -> PlanResponse:
    """
    Send a user message to the planner and receive its response.

    The graph runs synchronously inside a thread pool (P0-2.2 pattern) so the
    async FastAPI handler is not blocked.
    """
    import asyncio
    import concurrent.futures as cf

    config = _thread_config(request.thread_id)

    def _run() -> dict:
        g = _get_graph()
        result = {}
        for chunk in g.stream(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
            stream_mode="values",
        ):
            result = chunk
        return result

    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    last_message = _extract_last_ai_message(state)
    awaiting_approval = bool(state.get("planner_status") == "ready" and
                              not state.get("awaiting_user_clarification"))

    # Deserialise FinalPlan if present.
    plan_dict: Optional[Dict[str, Any]] = None
    raw_plan = state.get("final_plan")
    if raw_plan:
        try:
            plan_dict = FinalPlan(**raw_plan).model_dump()
        except Exception:
            plan_dict = raw_plan

    return PlanResponse(
        thread_id=request.thread_id,
        message=last_message,
        awaiting_clarification=bool(state.get("awaiting_user_clarification")),
        awaiting_approval=awaiting_approval,
        plan=plan_dict,
        planner_status=state.get("planner_status"),
    )


@app.post("/plan/{thread_id}/approve", response_model=PlanResponse)
async def approve_plan(thread_id: str, request: ApprovalRequest) -> PlanResponse:
    """
    Resume a plan that is suspended at the hitl_approval node.

    Send {"decision": "approved"} to cache and finish.
    Send {"decision": "edit", "feedback": "..."} to replan with feedback.
    Send {"decision": "cancelled"} to end the session.
    """
    import asyncio

    config = _thread_config(thread_id)
    resume_value = {"decision": request.decision, "feedback": request.feedback}

    def _run() -> dict:
        g = _get_graph()
        result = {}
        for chunk in g.stream(
            Command(resume=resume_value),
            config=config,
            stream_mode="values",
        ):
            result = chunk
        return result

    loop = asyncio.get_event_loop()
    try:
        state = await loop.run_in_executor(None, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    plan_dict: Optional[Dict[str, Any]] = None
    raw_plan = state.get("final_plan")
    if raw_plan:
        try:
            plan_dict = FinalPlan(**raw_plan).model_dump()
        except Exception:
            plan_dict = raw_plan

    return PlanResponse(
        thread_id=thread_id,
        message=_extract_last_ai_message(state),
        awaiting_clarification=bool(state.get("awaiting_user_clarification")),
        plan=plan_dict,
        planner_status=state.get("planner_status"),
    )
