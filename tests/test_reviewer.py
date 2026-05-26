"""
Tests for the reviewer agent.
No real LLM calls — ainvoke is mocked.
"""

from unittest.mock import patch, AsyncMock, MagicMock


class TestContentToText:

    def test_string_inputs(self):
        from src.agents.reviewer import _content_to_text
        assert _content_to_text("Hello world") == "Hello world"
        assert _content_to_text("") == ""
        assert _content_to_text(42) == "42"

    def test_list_inputs(self):
        from src.agents.reviewer import _content_to_text
        assert "Part one" in _content_to_text([{"text": "Part one"}, {"text": "Part two"}])
        assert "line a" in _content_to_text(["line a", "line b"])
        assert _content_to_text([]) == ""

    def test_dict_without_text_key_uses_str(self):
        from src.agents.reviewer import _content_to_text
        result = _content_to_text([{"type": "image", "url": "http://x"}])
        assert "image" in result or "url" in result


class TestReviewPlan:

    def test_empty_plan_skips_llm(self):
        from src.agents.reviewer import review_plan
        assert "No travel plan" in review_plan("")
        assert "No travel plan" in review_plan("   \n\t  ")

    def test_string_plan_calls_ainvoke(self):
        from src.agents.reviewer import review_plan
        fake_response = MagicMock()
        fake_response.content = "The plan looks solid."
        with patch("src.agents.reviewer.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(return_value=fake_response)
            mock_get_model.return_value = mock_model
            result = review_plan("Here is a Paris trip plan for 5 days.")
        assert result == "The plan looks solid."
        mock_model.ainvoke.assert_awaited_once()

    def test_list_content_handling(self):
        """List input is normalized before sending, list response is normalized after."""
        from src.agents.reviewer import review_plan
        fake_response = MagicMock()
        fake_response.content = [{"text": "Critique part 1"}, {"text": "Critique part 2"}]
        with patch("src.agents.reviewer.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(return_value=fake_response)
            mock_get_model.return_value = mock_model
            result = review_plan([{"text": "5 days in Tokyo with budget $2000."}])
        assert "Critique part 1" in result
        assert "Critique part 2" in result
        # Verify the input was normalized: user message should contain Tokyo
        _, call_args, _ = mock_model.ainvoke.mock_calls[0]
        assert "Tokyo" in str(call_args[0])

    def test_uses_reviewer_prompt_from_registry(self):
        from src.agents.reviewer import review_plan
        fake_response = MagicMock()
        fake_response.content = "OK"
        with patch("src.agents.reviewer.get_model") as mock_get_model, \
             patch("src.agents.reviewer.get_prompt", return_value="SYSTEM") as mock_prompt:
            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(return_value=fake_response)
            mock_get_model.return_value = mock_model
            review_plan("Trip plan content.")
        mock_prompt.assert_called_once_with("reviewer_prompt")
