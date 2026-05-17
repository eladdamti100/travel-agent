"""
Researcher agent — focused ReAct-style travel research agent.

The researcher is used for factual travel lookups, not full trip planning.
It can use travel database tools, cost tools, and web search when needed.

This agent is called by the LangGraph researcher node after the master
orchestrator routes an approved user request to the "research" path.
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.base import get_model
from src.graph.state import AgentState
from src.tools import ALL_TOOLS
from src.utils.logger import get_logger

logger = get_logger("researcher_agent")

_MAX_RESEARCH_STEPS = 6

_RESEARCHER_PROMPT = """You are Marco's travel researcher agent.

Your job is to answer focused travel research questions using tools.

You are NOT the full trip planner.
Use this agent for factual lookups such as:
- available flights
- available hotels
- available activities
- visa requirements
- cheapest flight or hotel
- available destinations
- simple travel cost facts
- real-time travel info if local database tools are insufficient

Rules:
1. Use tools for factual data. Do not invent prices, availability, visa rules, or database results.
2. Keep answers concise and structured.
3. If the user asks for flights and does not specify origin, use origin="TLV".
4. If the user asks for visa requirements and does not specify origin country, use origin_country="Israel".
5. If a tool returns no data, say so clearly.
6. Do not perform full itinerary planning here. If the request requires a full trip plan, it should have gone to cache_check/planner.
7. Never call the same tool with identical arguments more than once.
"""


_TOOL_BY_NAME = {tool.name: tool for tool in ALL_TOOLS}


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

    model = get_model(temperature=0, bind_tools=ALL_TOOLS)

    conversation = [
        SystemMessage(content=_RESEARCHER_PROMPT),
        HumanMessage(content=user_message),
    ]

    used_signatures: set[tuple[str, str]] = set()
    tool_call_count = state.get("tool_call_count", 0)

    for step in range(_MAX_RESEARCH_STEPS):
        response = model.invoke(conversation)
        conversation.append(response)

        tool_calls = getattr(response, "tool_calls", None)

        if not tool_calls:
            logger.info("Researcher completed after %d step(s).", step + 1)
            return {
                "messages": [response],
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


def research(query: str) -> str:
    """
    Compatibility helper for direct non-graph usage.

    Runs the researcher from a plain text query and returns the final text.
    """
    result = run_researcher({"messages": [HumanMessage(content=query)]})
    messages = result.get("messages", [])

    if not messages:
        return ""

    content = messages[-1].content
    return content if isinstance(content, str) else str(content)