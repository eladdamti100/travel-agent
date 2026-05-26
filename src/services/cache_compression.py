"""
Cache compression service.

Compresses full trip-plan answers into short bullet-point summaries
before storing them in the semantic cache.

The compressed version is stored alongside the original answer and can
be used for quick previews, logging, and token-efficient context injection
in future LLM calls.
"""

from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts.loader import get_prompt
from src.utils.logger import get_logger

logger = get_logger("cache_compression")

_COMPRESSION_USER_TEMPLATE = (
    "Summarize the following trip plan into 4–6 bullet points:\n\n{answer}"
)

_MAX_INPUT_CHARS = 6000
_MIN_COMPRESS_CHARS = 300


def compress_answer(answer: str) -> str:
    """
    Returns a compressed bullet-point summary of a full trip-plan answer.

    Falls back to a plain truncation if the LLM call fails, so cache
    storage is never blocked by a compression error.
    """
    if len(answer) < _MIN_COMPRESS_CHARS:
        logger.info("Cache compression skipped: answer is already short (%d chars).", len(answer))
        return answer

    from src.agents.base import get_model

    truncated = answer[:_MAX_INPUT_CHARS]

    try:
        model = get_model(temperature=0)
        response = model.invoke([
            SystemMessage(content=get_prompt("cache_compression_prompt")),
            HumanMessage(content=_COMPRESSION_USER_TEMPLATE.format(answer=truncated)),
        ])

        compressed = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        ).strip()

        logger.info(
            "Cache compression: %d chars → %d chars", len(answer), len(compressed)
        )

        return compressed

    except Exception as error:
        logger.warning("Cache compression failed, using truncation fallback: %s", error)
        return answer[:500] + "..." if len(answer) > 500 else answer
