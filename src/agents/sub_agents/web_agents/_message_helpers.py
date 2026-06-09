"""
Shared LangGraph message-stream helpers for the hierarchical web-agent team.

Used by TransportWebAgent and ExperienceWebAgent to extract individual tool
results from the message list returned by create_react_agent.ainvoke().
"""

from __future__ import annotations

from typing import Dict

from langchain_core.messages import AIMessage, ToolMessage


def extract_tool_results(
    messages: list,
    tool_name_to_key: Dict[str, str],
) -> Dict[str, str]:
    """
    Walk the message stream and map each ToolMessage content to a planner key.

    Strategy:
      1. Build a tool_call_id → tool_name index from every AIMessage's tool_calls.
      2. Use that index to annotate every ToolMessage with its originating tool name.
      3. Return only the keys present in *tool_name_to_key*.

    Args:
        messages: The ``response["messages"]`` list from create_react_agent.ainvoke.
        tool_name_to_key: Mapping from tool function name to raw_results dict key.

    Returns:
        Dict[str, str] ready to merge into PlannerToolResults.raw_results.
    """
    call_id_to_name: Dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                call_id_to_name[tc["id"]] = tc["name"]

    extracted: Dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_name = call_id_to_name.get(msg.tool_call_id, "")
            result_key = tool_name_to_key.get(tool_name)
            if result_key:
                extracted[result_key] = str(msg.content)

    return extracted
