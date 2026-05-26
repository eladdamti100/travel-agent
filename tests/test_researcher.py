"""
Tests for the researcher agent — run_researcher
LLM calls and tool invocations are fully mocked.
"""

from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage


def _make_state(content="What are the visa requirements for Japan?"):
    return {"messages": [HumanMessage(content=content)], "tool_call_count": 0}


def _ai_msg(content="Here is the answer.", tool_calls=None):
    return AIMessage(content=content, tool_calls=tool_calls or [])


class TestResearcher:

    def test_empty_state_returns_default_message(self):
        from src.agents.researcher import run_researcher
        result = run_researcher({"messages": [], "tool_call_count": 0})
        assert result["messages"]
        assert "travel research question" in result["messages"][0].content.lower()

    def test_clean_answer_returned_immediately(self):
        """Model returns a final answer with no tool calls → loop exits after 1 step."""
        from src.agents.researcher import run_researcher

        final = _ai_msg("Japan requires a visa for most nationalities.")

        with patch("src.agents.researcher.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.invoke.return_value = final
            mock_get_model.return_value = mock_model

            result = run_researcher(_make_state())

        assert result["messages"][0].content == "Japan requires a visa for most nationalities."
        mock_model.invoke.assert_called_once()

    def test_repeated_tool_call_stops_early(self):
        """Identical tool call issued twice → researcher stops with a loop-detection message."""
        from src.agents.researcher import run_researcher

        call = {"name": "check_visa", "args": {"country": "Japan"}, "id": "1"}
        repeated = _ai_msg(tool_calls=[call])

        fake_tool = MagicMock()
        fake_tool.name = "check_visa"
        fake_tool.invoke.return_value = "Visa required."

        with patch("src.agents.researcher.get_model") as mock_get_model, \
             patch("src.agents.researcher._TOOL_BY_NAME", {"check_visa": fake_tool}):
            mock_model = MagicMock()
            mock_model.invoke.return_value = repeated
            mock_get_model.return_value = mock_model

            result = run_researcher(_make_state())

        content = result["messages"][0].content
        assert "repeated" in content.lower() or "loop" in content.lower()

    def test_max_steps_reached_returns_limit_message(self):
        """Model always returns tool calls → researcher hits _MAX_RESEARCH_STEPS."""
        from src.agents.researcher import run_researcher, _MAX_RESEARCH_STEPS

        calls = [
            {"name": f"tool_{i}", "args": {"step": i}, "id": str(i)}
            for i in range(_MAX_RESEARCH_STEPS + 2)
        ]

        fake_tools = {
            f"tool_{i}": MagicMock(name=f"tool_{i}", invoke=MagicMock(return_value="result"))
            for i in range(_MAX_RESEARCH_STEPS + 2)
        }

        step = [0]

        def _invoke(msgs):
            call = calls[step[0]]
            step[0] += 1
            return _ai_msg(tool_calls=[call])

        with patch("src.agents.researcher.get_model") as mock_get_model, \
             patch("src.agents.researcher._TOOL_BY_NAME", fake_tools):
            mock_model = MagicMock()
            mock_model.invoke.side_effect = _invoke
            mock_get_model.return_value = mock_model

            result = run_researcher(_make_state())

        content = result["messages"][0].content
        assert "limit" in content.lower() or "step" in content.lower() or "narrow" in content.lower()
