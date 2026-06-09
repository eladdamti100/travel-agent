"""
Hierarchical Web Agent team — four autonomous sub-agents that sit below the
WebSupervisor and perform domain-specific live research.

  TransportWebAgent  — flights, ground transport, visa updates, geocoding
  StayWebAgent       — hotel reviews and live pricing trends
  ExperienceWebAgent — live events and local brewery scene
  ManagerWebAgent    — currency conversion, country metadata, Tavily research
"""

from src.agents.sub_agents.web_agents.experience_web_agent import ExperienceWebAgent
from src.agents.sub_agents.web_agents.manager_web_agent import ManagerWebAgent
from src.agents.sub_agents.web_agents.stay_web_agent import StayWebAgent
from src.agents.sub_agents.web_agents.transport_web_agent import TransportWebAgent

__all__ = [
    "TransportWebAgent",
    "StayWebAgent",
    "ExperienceWebAgent",
    "ManagerWebAgent",
]
