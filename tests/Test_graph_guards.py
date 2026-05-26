"""
Tests for detect_repetition — graph_guards.py
"""

from langchain_core.messages import AIMessage, HumanMessage


class TestDetectRepetition:

    def test_no_repetition_without_tool_calls(self):
        from src.utils.graph_guards import detect_repetition
        assert detect_repetition([]) is False
        assert detect_repetition([HumanMessage(content="hi"), AIMessage(content="hello")]) is False

    def test_unique_tool_calls_not_repetition(self):
        from src.utils.graph_guards import detect_repetition
        msg1 = AIMessage(content="", tool_calls=[{"name": "fetch_flights", "args": {"city": "Paris"}, "id": "1"}])
        msg2 = AIMessage(content="", tool_calls=[{"name": "fetch_hotels", "args": {"city": "Paris"}, "id": "2"}])
        assert detect_repetition([msg1, msg2]) is False

    def test_identical_call_detected(self):
        from src.utils.graph_guards import detect_repetition
        call = {"name": "fetch_flights", "args": {"city": "Paris"}, "id": "1"}
        msg1 = AIMessage(content="", tool_calls=[call])
        msg2 = AIMessage(content="", tool_calls=[call])
        assert detect_repetition([msg1, msg2]) is True

    def test_same_name_different_args_not_repetition(self):
        from src.utils.graph_guards import detect_repetition
        msg1 = AIMessage(content="", tool_calls=[{"name": "fetch_flights", "args": {"city": "Paris"}, "id": "1"}])
        msg2 = AIMessage(content="", tool_calls=[{"name": "fetch_flights", "args": {"city": "Tokyo"}, "id": "2"}])
        assert detect_repetition([msg1, msg2]) is False
