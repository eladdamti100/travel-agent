"""
LangGraph node implementations for every step in the travel planner graph.
"""

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.cache_checker import run_cache_check
from src.agents.cache_store import run_cache_store
from src.agents.context_enricher import extract_trip_context_deterministic
from src.agents.critic import critique_plan
from src.agents.master_orchestrator import run_master_orchestrator
from src.agents.planner import run_master_planner
from src.agents.preferences_memory_agent import run_preferences_memory
from src.agents.researcher import run_researcher
from src.agents.base import get_model as _get_unbound_model
from src.agents.validator import InputValidator, ai_validate, validate_input
from src.agents.vector_guard import is_vector_threat
from src.config.city_registry import CITY_KEYWORDS as _CITY_MAP
from src.config.settings import settings
from src.graph.state import AgentState
from src.models.trip_context import TripContext
from src.prompts.loader import get_prompt
from src.utils.logger import get_logger
from src.utils.modification_detector import detect_modification_context

logger = get_logger("nodes")

MAX_CRITIC_ATTEMPTS: int = settings.max_critic_attempts


# ── Node 1: Metadata Extraction ──────────────────────────────────────────────

def extract_metadata(state: AgentState) -> dict:
    """
    Lightweight preprocessing node that runs before validation.

    Resets all per-turn output fields so stale data from a previous plan
    cannot bleed into the current turn.  Does not route intent.
    """
    messages = state.get("messages", [])
    updates: dict = {
        # ── turn-level resets ────────────────────────────────────────────────
        "tool_call_count": 0,
        "force_replan": False,
        "critic_attempts": 0,
        "hitl_decision": "",
        "hitl_feedback": "",
        "hitl_edit_attempts": 0,
        # orchestrator
        "orchestrator_route": "",
        "orchestrator_reason": "",
        # cache
        "cache_status": "",
        "cache_answer": "",
        "cache_matched_query": "",
        "cache_similarity_score": 0.0,
        # planner outputs
        "planner_status": "",
        "planner_task_results": {},
        "planner_structured_results": {},
        "planner_dependency_graph": {},
        "planner_scheduler_result": {},
        "context_enrichment_status": "",
        "used_web_source": False,
    }

    if not messages:
        return updates

    last_content = getattr(messages[-1], "content", "")
    last_content_lower = last_content.lower()

    # Position-aware city detection: prefer the city after "to ".
    city_hits = [
        (last_content_lower.find(keyword), city)
        for keyword, city in _CITY_MAP.items()
        if keyword in last_content_lower
    ]
    if city_hits:
        to_idx = last_content_lower.rfind("to ")
        after_to = sorted(
            (pos, city) for pos, city in city_hits if to_idx != -1 and pos >= to_idx
        )
        updates["current_city"] = after_to[0][1] if after_to else sorted(city_hits)[0][1]
        logger.info("extract_metadata. detected_city=%s", updates["current_city"])

    budget_match = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", last_content_lower)
    if budget_match:
        updates["total_budget"] = float(budget_match.group(1).replace(",", ""))
        logger.info("extract_metadata. detected_budget=%.2f", updates["total_budget"])

    if detect_modification_context(last_content):
        updates["force_replan"] = True
        logger.info("extract_metadata. modification_detected=True force_replan=True")

    return updates


# ── Node 2: Validator ────────────────────────────────────────────────────────

def run_validator(state: AgentState) -> dict:
    """
    Security guardrail node — validates every user message before orchestration.

    Three-stage fast path (fastest first):
      1. Instant regex — blocks harm/injection/off-topic without any LLM.
      2. Travel keyword fast-approve — skips the Groq call for obvious travel messages.
      3. Groq LLM — only for ambiguous messages (~200 ms).

    HITL turns use is_hitl=True: short factual answers (airport codes, nationalities,
    durations, budgets) are fast-approved; harm and injection checks still run.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"validation_status": "approved"}

    last_content = getattr(messages[-1], "content", "")
    is_hitl = bool(state.get("awaiting_user_clarification"))

    regex_result = validate_input(last_content, is_hitl=is_hitl)
    if not regex_result.approved:
        logger.info("validator. verdict=%s is_hitl=%s", regex_result.verdict, is_hitl)
        return {
            "validation_status": regex_result.verdict.lower(),
            "messages": [AIMessage(content=regex_result.rejection_message)],
        }

    if is_hitl:
        logger.info("validator. status=approved path=hitl_fast_approve")
        return {"validation_status": "approved"}

    if InputValidator.is_clearly_travel(last_content):
        if not is_vector_threat(last_content):
            logger.info("validator. status=approved path=travel_keyword")
            return {"validation_status": "approved"}
        logger.info("validator. vector_threat=True routing_to_llm=True")

    result = ai_validate(last_content)
    if result is None:
        logger.info("validator. status=approved path=llm_unavailable")
        return {"validation_status": "approved"}

    logger.info("validator. verdict=%s reason=%s", result.verdict, result.reason)

    if not result.approved:
        return {
            "validation_status": result.verdict.lower(),
            "messages": [AIMessage(content=result.rejection_message)],
        }

    return {"validation_status": "approved"}


# ── Node 3: HITL Resume Context ──────────────────────────────────────────────

def resume_hitl_context_node(state: AgentState) -> dict:
    """
    Resumes a previously interrupted HITL planning flow by merging the user's
    clarification reply into the saved pending TripContext.
    """
    pending_context = state.get("pending_trip_context", {}) or {}
    new_context = extract_trip_context_deterministic(state)

    merged = {
        **pending_context,
        **{k: v for k, v in new_context.model_dump().items() if v not in (None, "", [])},
    }

    logger.info(
        "resume_hitl_context. pending_fields=%s merged_fields=%s",
        list(pending_context.keys()),
        list(merged.keys()),
    )

    try:
        return {
            "trip_context": TripContext(**merged).model_dump(),
            "awaiting_user_clarification": False,
        }
    except Exception as exc:
        logger.error("resume_hitl_context. merge_failed=%s falling_back_to_pending", exc)
        return {
            "trip_context": pending_context,
            "awaiting_user_clarification": False,
        }


# ── Node 4: Master Orchestrator ──────────────────────────────────────────────

def master_orchestrator_node(state: AgentState) -> dict:
    return run_master_orchestrator(state)


# ── Node 5: Preferences Memory ───────────────────────────────────────────────

def preferences_memory_node(state: AgentState) -> dict:
    return run_preferences_memory(state)


# ── Node 6: Researcher ───────────────────────────────────────────────────────

def researcher_node(state: AgentState) -> dict:
    return run_researcher(state)


# ── Node 7: Cache Check ──────────────────────────────────────────────────────

def cache_check_node(state: AgentState) -> dict:
    return run_cache_check(state)


# ── Node 8: Master Planner ──────────────────────────────────────────────────

def master_planner_node(state: AgentState) -> dict:
    """
    Wrapper with a hard wall-clock timeout so a stalled planner never blocks
    the graph indefinitely. The planner runs in a ThreadPoolExecutor (P0-2.2),
    so we enforce the timeout at this node boundary.
    """
    import concurrent.futures as _cf
    timeout = settings.planner_timeout_seconds
    with _cf.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_master_planner, state)
        try:
            return future.result(timeout=timeout)
        except _cf.TimeoutError:
            logger.error(
                "master_planner_node. status=timeout timeout_seconds=%.0f", timeout
            )
            future.cancel()
            return {
                "planner_status": "timeout",
                "messages": [
                    AIMessage(
                        content=(
                            "The planner took too long to respond. "
                            "Please try again with a simpler request, "
                            "or check that your API keys are configured correctly."
                        )
                    )
                ],
            }


# ── Node 9: Critic ───────────────────────────────────────────────────────────

def critic_node(state: AgentState) -> dict:
    """
    Runs the deterministic plan critic and stores the result in state.
    route_after_critic in router.py decides whether to replan or proceed.
    """
    result = critique_plan(state)

    attempts = (state.get("critic_attempts") or 0) + 1

    logger.info(
        "critic. passed=%s score=%d attempt=%d/%d",
        result.passed,
        result.score,
        attempts,
        MAX_CRITIC_ATTEMPTS,
    )

    return {
        "critic_attempts": attempts,
        "critique_result": {
            "passed": result.passed,
            "score": result.score,
            "reason": result.reason,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "completeness": result.completeness,
            "budget": {
                "budget": result.budget.budget,
                "flight_cost": result.budget.flight_cost,
                "hotel_cost": result.budget.hotel_cost,
                "activities_cost": result.budget.activities_cost,
                "total_estimated": result.budget.total_estimated,
                "overage": result.budget.overage,
                "within_budget": result.budget.within_budget,
            },
        },
    }


# ── Node 10: HITL Approval ────────────────────────────────────────────────────

def hitl_approval_node(state: AgentState) -> dict:
    """
    Suspends the graph so the user can approve, edit, or cancel the plan.

    LangGraph's interrupt() persists state to SqliteSaver and raises
    GraphInterrupt. main.py catches it, prompts the user, then resumes via
    graph.invoke(Command(resume=decision), config).

    Resume value: {"decision": "approved"|"edit"|"cancelled", "feedback": "<text>"}
    """
    from langgraph.types import interrupt
    from src.graph.router import MAX_HITL_EDIT_ATTEMPTS

    logger.info("hitl_approval. status=suspended")

    user_response = interrupt({
        "type": "plan_approval",
        "critique": state.get("critique_result", {}),
    })

    decision = user_response.get("decision", "approved") if isinstance(user_response, dict) else "approved"
    feedback = user_response.get("feedback", "") if isinstance(user_response, dict) else ""

    edit_attempts = (state.get("hitl_edit_attempts") or 0)
    if decision == "edit":
        edit_attempts += 1

    logger.info("hitl_approval. decision=%s edit_attempts=%d", decision, edit_attempts)

    if decision == "edit" and edit_attempts > MAX_HITL_EDIT_ATTEMPTS:
        logger.warning(
            "hitl_approval. edit_cap_reached=%d/%d ending_session",
            edit_attempts, MAX_HITL_EDIT_ATTEMPTS,
        )
        return {
            "hitl_decision": "cancelled",
            "hitl_feedback": "",
            "force_replan": False,
            "hitl_edit_attempts": edit_attempts,
            "messages": [
                AIMessage(
                    content=(
                        f"You've requested {edit_attempts} edits on this plan. "
                        "To keep things moving I'm ending this session — "
                        "please start a new conversation to plan a fresh trip."
                    )
                )
            ],
        }

    return {
        "hitl_decision": decision,
        "hitl_feedback": feedback,
        "force_replan": decision == "edit",
        "hitl_edit_attempts": edit_attempts,
    }


# ── Node 11: Cache Store ─────────────────────────────────────────────────────

def cache_store_node(state: AgentState) -> dict:
    return run_cache_store(state)


# ── Node 12: Summarizer ──────────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> dict:
    """
    Compact memory node.  Summarizes older messages into conversation_summary
    when history exceeds 10 messages so future LLM calls stay concise.
    Uses an unbound model (no tools) — the summarizer never calls tools.
    """
    messages = state.get("messages", [])

    if len(messages) <= 10:
        return {}

    to_summarize = messages[:-4]

    transcript_lines = []
    for msg in to_summarize:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        transcript_lines.append(f"{role}: {content[:300]}")

    if not transcript_lines:
        return {}

    model = _get_unbound_model(temperature=0)  # unbound — summarizer never calls tools

    summary_response = model.invoke([
        SystemMessage(content=get_prompt("summarizer_prompt")),
        HumanMessage(content="\n".join(transcript_lines)),
    ])

    summary = (
        summary_response.content
        if isinstance(summary_response.content, str)
        else str(summary_response.content)
    )

    logger.info(
        "summarizer. compressed_messages=%d summary_chars=%d",
        len(to_summarize),
        len(summary),
    )

    return {"conversation_summary": summary}
