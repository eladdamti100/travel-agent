from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import get_model
from src.graph.state import AgentState
from src.models.routing import RouteDecision, RouteType
from src.utils.logger import get_logger

logger = get_logger("master_orchestrator")


_ORCHESTRATOR_PROMPT = """You are the master orchestrator for Marco, a travel planning AI assistant.

Your job is to route an already-approved user message to exactly one high-level path.

Available routes:

1. preferences_memory
Use this for default user-memory conversations:
- The user asks what Marco remembers about their travel preferences.
- The user asks about saved profile details.
- The user states or updates a travel preference that should be saved.

Examples:
- "What do you remember about me?"
- "What are my travel preferences?"
- "Do you remember which airline I prefer?"
- "What food preference did I save?"
- "I prefer El Al"
- "I eat kosher"
- "I am vegan"
- "We are 4 travelers"
- "I prefer direct flights"
- "I like 5-star hotels"
- "I always want window seats"

2. research
Use this when the user asks for a direct factual lookup or learning/research answer
about available travel data, without asking for a complete trip plan.

Examples:
- "What hotels are available in Paris?"
- "Show me flights to Tokyo"
- "List activities in London"
- "What visa do I need for Japan?"
- "What destinations are available from TLV?"
- "What is the cheapest hotel in Berlin?"

3. cache_check
Use this for trip planning, itinerary building, cost planning, recommendations,
or any request that may require a full answer and should first be checked against cached previous answers.

Examples:
- "Plan me a 5-day trip to Paris"
- "Build an itinerary for London"
- "I want a trip to Tokyo with hotels and activities"
- "Create a budget vacation plan"
- "Plan the best family trip to New York under $4000"

Return only the structured route decision.
"""


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
            SystemMessage(content=_ORCHESTRATOR_PROMPT),
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