"""
Task registry for planner execution.

Maps PlannerTaskType values to the sub-agent responsible for producing them.

This is the first step toward scheduler-driven execution:
PlannerTaskType → executor owner → async execution.
"""

from typing import Dict, Iterable, List, Set

from src.agents.sub_agents.experience_agent import ExperienceAgent
from src.agents.sub_agents.stay_agent import StayAgent
from src.agents.sub_agents.transport_agent import TransportAgent
from src.models.planner import PlannerTaskType
from src.utils.logger import get_logger

logger = get_logger("task_registry")


_AGENT_FACTORIES = [
    TransportAgent,
    StayAgent,
    ExperienceAgent,
]


def get_planner_agents():
    """
    Returns all planner sub-agents.
    """
    return [factory() for factory in _AGENT_FACTORIES]


def get_task_to_agent_map() -> Dict[str, object]:
    """
    Builds a mapping from planner task key to the sub-agent that can produce it.
    """
    mapping: Dict[str, object] = {}

    for agent in get_planner_agents():
        for key in agent.result_keys:
            mapping[key] = agent

    logger.info(
        "Task registry built. tasks=%s",
        sorted(mapping.keys()),
    )

    return mapping


def get_agents_for_tasks(task_keys: Iterable[str]):
    """
    Returns unique sub-agents capable of producing at least one requested task.
    """
    requested: Set[str] = set(task_keys)
    agents = []

    for agent in get_planner_agents():
        if any(key in requested for key in agent.result_keys):
            agents.append(agent)

    logger.info(
        "Task registry selected agents=%s for tasks=%s",
        [
            getattr(agent, "agent_name", agent.__class__.__name__)
            for agent in agents
        ],
        sorted(requested),
    )

    return agents


def planner_task_values(tasks: Iterable[PlannerTaskType]) -> List[str]:
    """
    Converts PlannerTaskType values into their string task keys.
    """
    return [task.value for task in tasks]