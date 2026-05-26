"""
Reviewer agent — self-correction and quality control.

After the planner produces a travel plan, the reviewer critiques it
for completeness, budget realism, and missing information.
Called from main.py via the 'review' command, not as a graph node.
"""

import asyncio
from typing import Any

from langchain_core.messages import SystemMessage

from src.agents.base import get_model
from src.prompts.loader import get_prompt


def _content_to_text(content) -> str:
    """
    Normalize LangChain/Gemini response content into plain text.
    Gemini may return either a string or a list of content blocks.
    """
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )

    return str(content)


async def _review_plan_async(plan: str) -> str:
    """
    Async implementation — critiques a travel plan using ainvoke so the
    LLM call does not block the calling thread.
    """
    model = get_model(temperature=0.3)

    response = await model.ainvoke([
        SystemMessage(content=get_prompt("reviewer_prompt")),
        ("user", f"Please review this travel plan:\n\n{plan}"),
    ])

    return _content_to_text(response.content)


def review_plan(plan: Any) -> str:
    """
    Synchronous wrapper for the async reviewer.

    Accepts raw LangChain message content (str or list of content blocks)
    and normalizes it before review, so callers never need to do it themselves.
    Returns the reviewer's structured feedback as a plain string.
    """
    text = _content_to_text(plan)

    if not text.strip():
        return "No travel plan was provided for review."

    return asyncio.run(_review_plan_async(text))