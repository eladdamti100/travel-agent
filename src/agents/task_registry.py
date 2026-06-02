"""
Task registry for planner execution.

Maps PlannerTaskType values to the sub-agent responsible for producing them.
Features defensive dual-format normalization supporting both string keys and Enum objects.
"""

from typing import Dict, Iterable, List, Set, Any

from src.agents.sub_agents.experience_agent import ExperienceAgent
from src.agents.sub_agents.stay_agent import StayAgent
from src.agents.sub_agents.transport_agent import TransportAgent
from src.agents.sub_agents.web_agent import WebAgent
from src.models.planner import PlannerTaskType
from src.utils.logger import get_logger

logger = get_logger("task_registry")


_AGENT_FACTORIES = [
    TransportAgent,
    StayAgent,
    ExperienceAgent,
    WebAgent,
]


def get_planner_agents():
    """
    Returns all planner sub-agents.
    """
    return [factory() for factory in _AGENT_FACTORIES]


def get_task_to_agent_map() -> Dict[Any, object]:
    """
    Builds a robust, dual-format mapping from planner task keys to sub-agents.
    Defensively registers both string representations and native Enum objects to prevent KeyErrors.
    """
    mapping: Dict[Any, object] = {}

    for agent in get_planner_agents():
        for key in agent.result_keys:
            # Extract the canonical string value safely
            str_key = key.value if hasattr(key, "value") else str(key)
            
            # Explicitly map the string form
            mapping[str_key] = agent
            
            # Explicitly map the native Enum object form to guarantee runtime interoperability
            try:
                enum_key = PlannerTaskType(str_key)
                mapping[enum_key] = agent
            except ValueError:
                # If a custom task component is supplied outside the core Enum space, map as-is
                mapping[key] = agent

    logger.info(
        "Task registry built with dual-format normalization. Registered total accessible key variants: %d",
        len(mapping),
    )

    return mapping


def get_agents_for_tasks(task_keys: Iterable[Any]):
    """
    Returns unique sub-agents capable of producing at least one requested task.
    Transparently supports mixes of raw string task keys and PlannerTaskType Enums.
    """
    # Normalize all requested task keys to strings for unified set intersection checks
    requested_strings = {
        k.value if hasattr(k, "value") else str(k) for k in task_keys
    }
    agents = []

    for agent in get_planner_agents():
        # Normalize agent capability arrays to strings
        agent_strings = {
            k.value if hasattr(k, "value") else str(k) for k in agent.result_keys
        }
        
        if agent_strings.intersection(requested_strings):
            agents.append(agent)

    logger.info(
        "Task registry selected agents=%s for requested tasks=%s",
        [getattr(agent, "agent_name", agent.__class__.__name__) for agent in agents],
        sorted(list(requested_strings)),
    )

    return agents


def planner_task_values(tasks: Iterable[PlannerTaskType]) -> List[str]:
    """
    Converts PlannerTaskType values into their string task keys defensively.
    """
    return [task.value if hasattr(task, "value") else str(task) for task in tasks]