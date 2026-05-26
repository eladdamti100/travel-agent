import re
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from src.agents.cache_checker import run_cache_check
from src.agents.cache_store import run_cache_store
from src.agents.context_enricher import (
    extract_trip_context_deterministic,
)
from src.agents.master_orchestrator import run_master_orchestrator
from src.agents.planner import run_master_planner
from src.agents.preferences_memory_agent import run_preferences_memory
from src.agents.researcher import run_researcher
from src.graph.state import AgentState
from src.models.trip_context import TripContext
from src.tools import ALL_TOOLS
from src.utils.logger import get_logger
from src.prompts.loader import get_prompt

logger = get_logger("nodes")


_RETRY_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)

# City keyword → canonical city name.
# This is only lightweight metadata extraction, not routing.
_CITY_MAP = {
    "paris": "Paris",
    "london": "London",
    "tokyo": "Tokyo",
    "new york": "New York",
    "berlin": "Berlin",
}

# Lazy-initialised bound model for the legacy planner path.
_model = None


def _get_model():
    """
    Returns the main tool-bound planning model.

    This is still used by the legacy planner path and summarizer.
    The new researcher, preferences-memory, cache, and master-planner paths live
    in their own agent files.
    """
    global _model

    if _model is None:
        from src.agents.base import get_model

        _model = get_model(temperature=0, bind_tools=ALL_TOOLS)

    return _model


# ── Node 1: Metadata Extraction ──────────────────────────────────────────────

def extract_metadata(state: AgentState) -> dict:
    """
    Lightweight preprocessing node that runs before validation.

    This node does not route intent.
    It only:
      - resets tool_call_count for the current turn
      - detects supported destination city
      - detects a USD budget if explicitly written as "$1500"
    """
    messages = state.get("messages", [])
    updates: dict = {"tool_call_count": 0}

    if not messages:
        return updates

    last_content = getattr(messages[-1], "content", "").lower()

    for keyword, city in _CITY_MAP.items():
        if keyword in last_content:
            updates["current_city"] = city
            logger.info("Detected city: %s", city)
            break

    budget_match = re.search(r"\$(\d[\d,]*(?:\.\d+)?)", last_content)
    if budget_match:
        budget = float(budget_match.group(1).replace(",", ""))
        updates["total_budget"] = budget
        logger.info("Detected budget: $%s", budget)

    return updates


# ── Node 2: Validator ────────────────────────────────────────────────────────

def run_validator(state: AgentState) -> dict:
    """
    Security guardrail node — validates every user message before orchestration.

    Validation order:
      1. AI validator via Groq, if available.
      2. Rule-based fallback validator.

    If blocked, the rejection message is added to State and the graph ends.
    If approved, validation_status is set to "approved".
    """
    from src.agents.ai_validator import ai_validate
    from src.agents.validator import validate_input

    messages = state.get("messages", [])
    if not messages:
        return {"validation_status": "approved"}

    # HITL replies are factual travel details answering the planner's question.
    # Running the full validator on them causes false positives (e.g. "Israeli passport").
    if state.get("awaiting_user_clarification"):
        return {"validation_status": "approved"}

    last_content = getattr(messages[-1], "content", "")

    result = ai_validate(last_content)

    if result is None:
        logger.info("Validator: using rule-based fallback.")
        result = validate_input(last_content)

    logger.info(
        "Validator: verdict=%s reason=%s",
        result.verdict,
        result.reason,
    )

    if not result.approved:
        return {
            "validation_status": result.verdict.lower(),
            "messages": [AIMessage(content=result.rejection_message)],
        }

    return {"validation_status": "approved"}


# ── Node 3: HITL Resume Context ──────────────────────────────────────────────

def resume_hitl_context_node(state: AgentState) -> dict:
    """
    Resumes a previously interrupted HITL planning flow.

    This node:
      1. Loads the pending TripContext from State.
      2. Extracts new structured fields from the latest user clarification.
      3. Merges the new fields into the pending context.
      4. Returns the merged context so master_planner can continue planning.

    This bypasses:
      - master_orchestrator
      - semantic cache
      - researcher routing

    because the user is continuing an already-started planning flow.
    """
    pending_context = state.get("pending_trip_context", {}) or {}

    new_context = extract_trip_context_deterministic(state)

    merged = {
        **pending_context,
        **{
            key: value
            for key, value in new_context.model_dump().items()
            if value not in (None, "", [])
        },
    }

    logger.info(
        "HITL resume merged context. pending=%s merged=%s",
        pending_context,
        merged,
    )

    return {
        "trip_context": TripContext(**merged).model_dump(),
        "awaiting_user_clarification": False,
    }


# ── Node 4: Master Orchestrator ──────────────────────────────────────────────

def master_orchestrator_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the master orchestrator.
    """
    return run_master_orchestrator(state)


# ── Node 5: Preferences Memory ───────────────────────────────────────────────

def preferences_memory_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the preferences memory agent.

    The preferences memory agent internally decides whether to:
      - recall saved user preferences
      - update saved user preferences
    """
    return run_preferences_memory(state)


# ── Node 6: Researcher ───────────────────────────────────────────────────────

def researcher_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the researcher agent.
    """
    return run_researcher(state)


# ── Node 7: Cache Check ──────────────────────────────────────────────────────

def cache_check_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the semantic cache checker.
    """
    return run_cache_check(state)


# ── Node 8: Master Planner ──────────────────────────────────────────────────

def master_planner_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for the master planner.

    This is the new planner path used after semantic cache miss.
    """
    return run_master_planner(state)


# ── Node 9: Legacy Planner Agent ─────────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """
    Legacy planner node.

    This is kept temporarily for backward compatibility until the graph fully
    routes cache_miss to master_planner instead of this legacy node.
    """
    profile_lines = []

    if state.get("preferred_airline"):
        profile_lines.append(f"- Preferred airline: {state['preferred_airline']}")

    if state.get("food_preference"):
        profile_lines.append(f"- Dietary preference: {state['food_preference']}")

    if state.get("num_travelers"):
        profile_lines.append(f"- Traveling with: {state['num_travelers']} people")

    if state.get("travel_preferences"):
        profile_lines.append(f"- Additional preferences:\n{state['travel_preferences']}")

    summary = state.get("conversation_summary", "")
    planner_prompt = get_prompt("planner_prompt")
    
    if profile_lines or summary:
        extra = ""

        if profile_lines:
            extra += (
                "\n\n## User Profile (remembered from previous sessions)\n"
                + "\n".join(profile_lines)
                + "\nAlways apply these preferences when recommending flights, hotels, and activities."
            )

        if summary:
            extra += f"\n\n## Conversation Summary (past context)\n{summary}"

        system_msg = SystemMessage(content=planner_prompt + extra)

    else:
        system_msg = SystemMessage(content=planner_prompt)
        
    all_messages = state["messages"]

    if summary and len(all_messages) > 10:
        messages_to_send = all_messages[-8:]
        logger.info(
            "Compact memory: sending %d/%d messages to LLM.",
            len(messages_to_send),
            len(all_messages),
        )
    else:
        messages_to_send = all_messages

    messages = [system_msg] + messages_to_send

    max_retries = 2

    for attempt in range(max_retries):
        try:
            response = _get_model().invoke(messages)
            break

        except Exception as error:
            err = str(error)

            if (
                ("429" in err or "RESOURCE_EXHAUSTED" in err)
                and attempt < max_retries - 1
            ):
                match = _RETRY_PATTERN.search(err)
                wait = int(float(match.group(1))) + 3 if match else 30

                logger.warning(
                    "Rate limited — waiting %ss. attempt=%d/%d",
                    wait,
                    attempt + 1,
                    max_retries,
                )

                time.sleep(wait)

            else:
                raise

    count = state.get("tool_call_count", 0)

    if hasattr(response, "tool_calls") and response.tool_calls:
        count += len(response.tool_calls)
        names = [tool_call["name"] for tool_call in response.tool_calls]

        logger.info(
            "Tool calls +%d total=%d names=%s",
            len(response.tool_calls),
            count,
            names,
        )

    return {
        "messages": [response],
        "tool_call_count": count,
    }


# ── Node 10: Circuit Breaker ─────────────────────────────────────────────────

def circuit_breaker(state: AgentState) -> dict:
    """
    Safety node — fires when the agent exceeds MAX_TOOL_CALLS or enters a
    repetitive tool-call loop.
    """
    count = state.get("tool_call_count", 0)

    logger.warning("Circuit breaker triggered after %d tool calls.", count)

    return {
        "messages": [
            AIMessage(
                content=(
                    "I've reached my processing limit for this request. "
                    "This usually means the destination or route isn't in my database, "
                    "or the question is ambiguous. Please try rephrasing, or ask about "
                    "a supported city: Paris, London, Tokyo, New York, or Berlin."
                )
            )
        ]
    }


# ── Node 11: Reviewer ────────────────────────────────────────────────────────

def reviewer_node(state: AgentState) -> dict:
    """
    Quality-control node — automatically critiques Marco's final travel plan.

    This remains a node-level wrapper until reviewer is moved to its own
    run_reviewer(...) agent module.
    """
    from src.agents.reviewer import review_plan

    last_msg = state["messages"][-1]
    content = last_msg.content

    if isinstance(content, list):
        content = "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )

    review = review_plan(str(content))

    logger.info("Reviewer node completed critique.")

    return {
        "messages": [
            AIMessage(content=f"\n---\n**Plan Review (auto):**\n{review}")
        ]
    }


# ── Node 12: Cache Store ─────────────────────────────────────────────────────

def cache_store_node(state: AgentState) -> dict:
    """
    LangGraph node wrapper for storing successful answers in semantic cache.
    """
    return run_cache_store(state)


# ── Node 13: Summarizer ──────────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> dict:
    """
    Compact memory node.

    When message history exceeds 10 messages, it summarizes older messages into
    conversation_summary and future LLM calls use only recent messages plus the summary.
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

    summary_response = _get_model().invoke([
        SystemMessage(content="You are a concise conversation summarizer."),
        HumanMessage(
            content=(
                "Summarize this travel planning conversation in 3–4 bullet points. "
                "Include: destinations discussed, budgets, user preferences, and key decisions.\n\n"
                + "\n".join(transcript_lines)
            )
        ),
    ])

    summary = (
        summary_response.content
        if isinstance(summary_response.content, str)
        else str(summary_response.content)
    )

    logger.info(
        "Summarizer: compressed %d messages into %d chars.",
        len(to_summarize),
        len(summary),
    )

    return {"conversation_summary": summary}


# ── Tool Node ────────────────────────────────────────────────────────────────

def build_tools_node():
    """
    Builds the generic LangGraph ToolNode for the legacy planner path.
    """
    return ToolNode(ALL_TOOLS)