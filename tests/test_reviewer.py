"""
Tests for the reviewer agent.
No real LLM calls — ainvoke is mocked.
"""

from unittest.mock import patch, AsyncMock, MagicMock


def test_content_to_text():
    from src.agents.reviewer import _content_to_text
    assert _content_to_text("Hello world") == "Hello world"
    assert _content_to_text("") == ""
    assert _content_to_text(42) == "42"
    assert "Part one" in _content_to_text([{"text": "Part one"}, {"text": "Part two"}])
    assert "line a" in _content_to_text(["line a", "line b"])
    assert _content_to_text([]) == ""
    result = _content_to_text([{"type": "image", "url": "http://x"}])
    assert "image" in result or "url" in result


def test_review_plan_empty():
    from src.agents.reviewer import review_plan
    assert "No travel plan" in review_plan("")
    assert "No travel plan" in review_plan("   \n\t  ")


def test_review_plan_with_llm():
    from src.agents.reviewer import review_plan

    def _mock(content):
        m = MagicMock()
        m.ainvoke = AsyncMock(return_value=MagicMock(content=content))
        return m

    # String plan + string response
    with patch("src.agents.reviewer.get_model") as mock_get, \
         patch("src.agents.reviewer.get_prompt", return_value="SYSTEM") as mock_prompt:
        mock_get.return_value = _mock("The plan looks solid.")
        result = review_plan("Here is a Paris trip plan for 5 days.")
    assert result == "The plan looks solid."
    mock_prompt.assert_called_once_with("reviewer_prompt")

    # List plan + list response
    with patch("src.agents.reviewer.get_model") as mock_get:
        mock_model = _mock([{"text": "Critique part 1"}, {"text": "Critique part 2"}])
        mock_get.return_value = mock_model
        result = review_plan([{"text": "5 days in Tokyo with budget $2000."}])
    assert "Critique part 1" in result and "Critique part 2" in result
    _, call_args, _ = mock_model.ainvoke.mock_calls[0]
    assert "Tokyo" in str(call_args[0])
