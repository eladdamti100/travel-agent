"""
Researcher agent — focused ReAct-style travel research agent.

The researcher is used for factual travel lookups, not full trip planning.
It can use travel database tools, cost tools, and web search when needed.

This agent is called by the LangGraph researcher node after the master
orchestrator routes an approved user request to the "research" path.
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.base import get_model
from src.agents.cyber_agent import CyberAgent
from src.graph.state import AgentState
from src.prompts.loader import get_prompt
from src.tools import ALL_TOOLS
from src.utils.logger import get_logger
from src.utils.token_tracker import log_token_usage

logger = get_logger("researcher_agent")

_MAX_RESEARCH_STEPS = 6
_SECURITY_TIMEOUT = 5.0

_TOOL_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}
_cyber = CyberAgent()
_URL_RE = re.compile(r"https?://\S{10,}", re.ASCII)


def _run_async_safely(coro, *, default=None):
    """Run an async CyberAgent coroutine safely from synchronous context."""
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=_SECURITY_TIMEOUT)
    except Exception as exc:
        logger.warning("researcher.security_check_error. error=%s using=default", exc)
        return default


def _secure_response_content(raw_content: str) -> str:
    """Apply PII redaction and malicious URL blocking to a final researcher response."""
    content = raw_content

    redacted = _run_async_safely(
        _cyber.redact_sensitive_data(content),
        default=content,
    )
    if redacted is not None:
        content = redacted

    urls = _URL_RE.findall(content)
    if urls:
        malicious = _run_async_safely(
            _cyber.check_urls(list(set(urls))),
            default=[],
        )
        for bad_url in (malicious or []):
            content = content.replace(bad_url, "[BLOCKED MALICIOUS URL]")

    return content


def run_researcher(state: AgentState) -> dict:
    """
    Runs a focused ReAct-style research flow.

    The agent receives the current user message, chooses relevant tools,
    receives tool results, and returns a final concise answer.
    """
    messages = state.get("messages", [])

    if not messages:
        return {
            "messages": [
                AIMessage(content="Please ask a specific travel research question.")
            ]
        }

    user_message = getattr(messages[-1], "content", "")

    # ── Inbound security: block prompt injection before touching the LLM ─────
    flagged = _run_async_safely(
        _cyber.check_prompt_injection(user_message),
        default=False,
    )
    if flagged:
        logger.warning(
            "researcher.security. action=blocked reason=prompt_injection "
            "message_len=%d",
            len(user_message),
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Security: your request was blocked because it triggered "
                        "a prompt-injection check. Please rephrase your question."
                    )
                )
            ],
            "tool_call_count": state.get("tool_call_count", 0),
        }

    model = get_model(temperature=0, bind_tools=ALL_TOOLS)

    conversation = [
        SystemMessage(content=get_prompt("researcher_prompt")),
        HumanMessage(content=user_message),
    ]

    used_signatures: set[tuple[str, str]] = set()
    tool_call_count = state.get("tool_call_count", 0)

    for step in range(_MAX_RESEARCH_STEPS):
        response = model.invoke(conversation)
        log_token_usage(response, call_site=f"researcher.step_{step}")
        conversation.append(response)

        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            logger.info("Researcher completed after %d step(s).", step + 1)
            # ── Outbound security: redact PII + block malicious URLs ──────
            secured_content = _secure_response_content(
                str(getattr(response, "content", ""))
            )
            return {
                "messages": [AIMessage(content=secured_content)],
                "tool_call_count": tool_call_count,
            }

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            signature = (tool_name or "", str(tool_args))

            if signature in used_signatures:
                logger.warning(
                    "Researcher stopped repeated tool call: %s %s",
                    tool_name,
                    tool_args,
                )
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I found that the same research tool was being requested repeatedly, "
                                "so I stopped the lookup to avoid a loop. Please rephrase the research question."
                            )
                        )
                    ],
                    "tool_call_count": tool_call_count,
                }

            used_signatures.add(signature)

            tool = _TOOL_BY_NAME.get(tool_name)
            if tool is None:
                tool_result = f"Error: unknown research tool '{tool_name}'."
            else:
                try:
                    logger.info("Researcher calling tool: %s args=%s", tool_name, tool_args)
                    tool_result = tool.invoke(tool_args)
                    tool_call_count += 1
                except Exception as error:
                    logger.error("Researcher tool failed: %s", error)
                    tool_result = f"Tool error: {error}"

            if isinstance(tool_result, str) and "No flights found" in tool_result:
                tool_result += (
                    "\nSupported origins in the local database currently include TLV."
                )

            conversation.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            )

    logger.warning("Researcher reached max research steps.")

    return {
        "messages": [
            AIMessage(
                content=(
                    "I reached the research step limit before producing a final answer. "
                    "Please ask a narrower travel research question."
                )
            )
        ],
        "tool_call_count": tool_call_count,
    }
