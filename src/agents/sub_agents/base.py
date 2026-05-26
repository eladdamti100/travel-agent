from abc import ABC, abstractmethod
from typing import Tuple

from src.models.planner import PlannerToolResults
from src.models.trip_context import TripContext


class BaseSubAgent(ABC):
    """
    Base interface for planner sub-agents.

    Important:
      Sub-agents must return their own independent PlannerToolResults.
      They must not mutate shared state, so they can safely run in parallel.
    """

    agent_name: str = "base_sub_agent"
    result_keys: Tuple[str, ...] = ()

    @abstractmethod
    async def run(
        self,
        *,
        context: TripContext,
    ) -> PlannerToolResults:
        """
        Executes the sub-agent logic and returns independent planner outputs.
        """
        raise NotImplementedError