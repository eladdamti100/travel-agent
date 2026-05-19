from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_model
from src.graph.state import AgentState
from src.models.routing import RouteDecision, RouteType
from src.utils.logger import get_logger
from src.prompts.loader import get_prompt

logger = get_logger("master_orchestrator")


def run_master_orchestrator(state: AgentState) -> dict:
    """
    Routes an approved user request to the next high-level execution path.

    The validator has already approved the message before this function runs.
    This function does not execute tools and does not answer the user directly.
    It only writes a structured routing decision into the graph state.
    """
    messages = state.get("messages", [])
    if not messages:
        return {
            "orchestrator_route": RouteType.CACHE_CHECK.value,
            "orchestrator_reason": "No user message was found, so the request defaults to cache check.",
        }

    last_content = getattr(messages[-1], "content", "")

    try:
        model = get_model(temperature=0).with_structured_output(RouteDecision)
        decision = model.invoke([
            SystemMessage(content=get_prompt("orchestrator_prompt")),
            HumanMessage(content=last_content),
        ])

        logger.info(
            "Master orchestrator: route=%s reason=%s",
            decision.route.value,
            decision.reason,
        )

        return {
            "orchestrator_route": decision.route.value,
            "orchestrator_reason": decision.reason,
        }

    except Exception as error:
        logger.error("Master orchestrator failed: %s", error)

        return {
            "orchestrator_route": RouteType.CACHE_CHECK.value,
            "orchestrator_reason": (
                "Master orchestrator failed, so the request safely defaults to cache check."
            ),
        }