"""
Reviewer agent — self-correction and quality control.

After the planner produces a travel plan, the reviewer critiques it
for completeness, budget realism, and missing information.
Called from main.py via the 'review' command, not as a graph node.
"""

from langchain_core.messages import SystemMessage
from src.agents.base import get_model
from src.prompts.loader import get_prompt

_model = get_model(temperature=0.3)


def _content_to_text(content) -> str:
    """
    Normalize LangChain/Gemini response content into plain text.
    Gemini may return either a string or a list of content blocks.
    """
    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )

    return str(content)


def review_plan(plan: str) -> str:
    """
    Critique and score a travel plan string.
    Returns the reviewer's structured feedback as a plain string.
    """
    if not plan or not plan.strip():
        return "No travel plan was provided for review."

    response = _model.invoke(
        [
            SystemMessage(content=get_prompt("reviewer_prompt")),
            ("user", f"Please review this travel plan:\n\n{plan}"),
        ]
    )

    return _content_to_text(response.content)