"""
Dispatch tracker — thread-safe shared state for monitoring active sub-agents.

Both DBSupervisor and WebSupervisor import from here to mark which agents are
running. The CLI status monitor in main.py reads this via get_active_dispatch_agents()
(re-exported from web_supervisor for backward compatibility).
"""

import threading

_dispatch_lock = threading.Lock()
_dispatch_active: set = set()


def get_active_dispatch_agents() -> list:
    """Returns a sorted snapshot of agent names currently running in dispatch."""
    with _dispatch_lock:
        return sorted(_dispatch_active)


def mark_agent_started(name: str) -> None:
    with _dispatch_lock:
        _dispatch_active.add(name)


def mark_agent_done(name: str) -> None:
    with _dispatch_lock:
        _dispatch_active.discard(name)
