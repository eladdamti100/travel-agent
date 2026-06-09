"""
Task registry for planner execution.

Maps PlannerTaskType values to the sub-agent responsible for producing them.
Features defensive dual-format normalization supporting both string keys and Enum objects.

Sub-agent tiers
---------------
  Tier 1 (DB-backed):  TransportAgent, StayAgent, ExperienceAgent
  Tier 2 (Web / LLM):  TransportWebAgent, StayWebAgent, ExperienceWebAgent, ManagerWebAgent
"""

from typing import Any, Dict, Iterable, List

from src.agents.sub_agents.experience_agent import ExperienceAgent
from src.agents.sub_agents.stay_agent import StayAgent
from src.agents.sub_agents.transport_agent import TransportAgent
from src.agents.sub_agents.web_agents.experience_web_agent import ExperienceWebAgent
from src.agents.sub_agents.web_agents.manager_web_agent import ManagerWebAgent
from src.agents.sub_agents.web_agents.stay_web_agent import StayWebAgent
from src.agents.sub_agents.web_agents.transport_web_agent import TransportWebAgent
from src.models.planner import PlannerTaskType
from src.utils.logger import get_logger

logger = get_logger("task_registry")

# ── Tier 1: DB-backed agents (fast, deterministic SQLite tools) ───────────────
_DB_AGENT_FACTORIES = [
    TransportAgent,    # fetch_flights, check_visa
    StayAgent,         # fetch_hotels
    ExperienceAgent,   # fetch_activities, fetch_restaurants, local_transport_guide,
                       # fetch_weather, events_finder, airport_transfer_info
]

# ── Tier 2: Hierarchical web agents (live APIs, LLM ReAct loops) ─────────────
_WEB_AGENT_FACTORIES = [
    TransportWebAgent,    # geocode_location, transport_live_research
    StayWebAgent,         # stay_live_research
    ExperienceWebAgent,   # fetch_live_events, fetch_breweries, experience_web_research
    ManagerWebAgent,      # live_currency_conversion, fetch_country_metadata, web_research_tavily
]

# ── Combined — backward compatibility ─────────────────────────────────────────
_AGENT_FACTORIES = _DB_AGENT_FACTORIES + _WEB_AGENT_FACTORIES


# ── Public constructors ───────────────────────────────────────────────────────

def get_db_agents():
    """Returns fresh Tier 1 DB-backed sub-agent instances."""
    return [factory() for factory in _DB_AGENT_FACTORIES]


def get_web_agents():
    """Returns fresh Tier 2 hierarchical web agent instances."""
    return [factory() for factory in _WEB_AGENT_FACTORIES]


def get_planner_agents():
    """Returns all planner sub-agents (Tier 1 + Tier 2). Backward compatible."""
    return [factory() for factory in _AGENT_FACTORIES]


# ── Task-filtered constructors ────────────────────────────────────────────────

def get_db_agents_for_tasks(task_keys: Iterable[Any]):
    """
    Returns Tier 1 DB sub-agents capable of producing at least one requested task.
    Accepts mixes of raw string keys and PlannerTaskType enums.
    """
    return _filter_agents(_DB_AGENT_FACTORIES, task_keys, tier="db")


def get_web_agents_for_tasks(task_keys: Iterable[Any]):
    """
    Returns Tier 2 web sub-agents capable of producing at least one requested task.
    Accepts mixes of raw string keys and PlannerTaskType enums.
    """
    return _filter_agents(_WEB_AGENT_FACTORIES, task_keys, tier="web")


def get_agents_for_tasks(task_keys: Iterable[Any]):
    """
    Returns unique sub-agents (all tiers) covering at least one requested task.
    Backward compatible — transparently supports mixes of raw strings and enums.
    """
    return _filter_agents(_AGENT_FACTORIES, task_keys, tier="all")


def _filter_agents(factories, task_keys: Iterable[Any], tier: str = "all"):
    """Filter a factory list to agents covering at least one requested task key."""
    requested_strings = {
        k.value if hasattr(k, "value") else str(k) for k in task_keys
    }
    agents = []
    for agent in [factory() for factory in factories]:
        agent_strings = {
            k.value if hasattr(k, "value") else str(k) for k in agent.result_keys
        }
        if agent_strings.intersection(requested_strings):
            agents.append(agent)

    logger.info(
        "Task registry [tier=%s] selected agents=%s for tasks=%s",
        tier,
        [getattr(a, "agent_name", a.__class__.__name__) for a in agents],
        sorted(list(requested_strings)),
    )
    return agents


# ── Dual-format task → agent map ─────────────────────────────────────────────

def get_task_to_agent_map() -> Dict[Any, object]:
    """
    Builds a robust, dual-format mapping from planner task keys to sub-agents.
    Defensively registers both string representations and native Enum objects to prevent KeyErrors.
    """
    mapping: Dict[Any, object] = {}

    for agent in get_planner_agents():
        for key in agent.result_keys:
            str_key = key.value if hasattr(key, "value") else str(key)
            mapping[str_key] = agent
            try:
                enum_key = PlannerTaskType(str_key)
                mapping[enum_key] = agent
            except ValueError:
                mapping[key] = agent

    logger.info(
        "Task registry built with dual-format normalization. "
        "Registered total accessible key variants: %d",
        len(mapping),
    )
    return mapping


def planner_task_values(tasks: Iterable[PlannerTaskType]) -> List[str]:
    """Converts PlannerTaskType values into their string task keys defensively."""
    return [task.value if hasattr(task, "value") else str(task) for task in tasks]
