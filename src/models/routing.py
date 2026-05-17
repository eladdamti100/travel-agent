from enum import Enum
from pydantic import BaseModel, Field


class RouteType(str, Enum):
    """
    High-level routing targets selected by the master orchestrator.
    """

    PREFERENCES_MEMORY = "preferences_memory"
    RESEARCH = "research"
    CACHE_CHECK = "cache_check"


class RouteDecision(BaseModel):
    """
    Structured routing decision returned by the master orchestrator.
    """

    route: RouteType = Field(
        description="The next high-level route for the approved user request."
    )
    reason: str = Field(
        description="A short explanation of why this route was selected."
    )