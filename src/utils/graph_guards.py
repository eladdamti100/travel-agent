"""
Repetition detection — guards against infinite tool-call loops.

The router (src/graph/router.py) calls detect_repetition() before each
agent step. If the agent tries to call the same tool with the same
arguments that it already called in this turn, we short-circuit to the
circuit_breaker node instead of letting the loop continue.
"""


def detect_repetition(messages: list) -> bool:
    """
    Return True if the agent has issued an identical tool call at least twice
    within the current message history.

    A tool call signature is (tool_name, str(args_dict)).
    Comparing as strings is intentional: it catches exact repetitions without
    requiring deep equality on argument types.
    """
    seen: set = set()
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            continue
        for tc in tool_calls:
            sig = (tc.get("name", ""), str(tc.get("args", {})))
            if sig in seen:
                return True
            seen.add(sig)
    return False
