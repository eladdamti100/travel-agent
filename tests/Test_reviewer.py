"""
Tests for the reviewer agent.
No real LLM calls — ainvoke is mocked.
"""

from unittest.mock import patch, AsyncMock, MagicMock


class TestContentToText:

    def test_plain_string_returned_unchanged(self):
        from src.agents.reviewer import _content_to_text
        assert _content_to_text("Hello world") == "Hello world"

    def test_empty_string_returned(self):
        from src.agents.reviewer import _content_to_text
        assert _content_to_text("") == ""

    def test_list_of_text_dicts_joined(self):
        from src.agents.reviewer import _content_to_text
        content = [{"text": "Part one"}, {"text": "Part two"}]
        result = _content_to_text(content)
        assert "Part one" in result
        assert "Part two" in result

    def test_list_of_strings_joined(self):
        from src.agents.reviewer import _content_to_text
        result = _content_to_text(["line a", "line b"])
        assert "line a" in result
        assert "line b" in result

    def test_list_dict_without_text_key_uses_str(self):
        from src.agents.reviewer import _content_to_text
        content = [{"type": "image", "url": "http://x"}]
        result = _content_to_text(content)
        assert "image" in result or "url" in result or "http" in result

    def test_empty_list_returns_empty_string(self):
        from src.agents.reviewer import _content_to_text
        assert _content_to_text([]) == ""

    def test_non_string_non_list_coerced_to_str(self):
        from src.agents.reviewer import _content_to_text
        assert _content_to_text(42) == "42"


class TestReviewPlan:

    def test_empty_string_returns_no_plan_message(self):
        from src.agents.reviewer import review_plan
        result = review_plan("")
        assert "No travel plan" in result

    def test_whitespace_only_returns_no_plan_message(self):
        from src.agents.reviewer import review_plan
        result = review_plan("   \n\t  ")
        assert "No travel plan" in result

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

    def test_list_content_normalized_before_review(self):
        """review_plan accepts a list of content blocks (Gemini format)."""
        from src.agents.reviewer import review_plan

        fake_response = MagicMock()
        fake_response.content = "Looks good."

        with patch("src.agents.reviewer.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(return_value=fake_response)
            mock_get_model.return_value = mock_model

            result = review_plan([{"text": "5 days in Tokyo with budget $2000."}])

        assert result == "Looks good."
        _, call_args, _ = mock_model.ainvoke.mock_calls[0]
        messages = call_args[0]
        user_msg = messages[-1]
        assert "Tokyo" in str(user_msg)

    def test_response_list_content_normalized(self):
        """When the LLM returns a list of blocks, _content_to_text normalizes it."""
        from src.agents.reviewer import review_plan

        fake_response = MagicMock()
        fake_response.content = [{"text": "Critique part 1"}, {"text": "Critique part 2"}]

        with patch("src.agents.reviewer.get_model") as mock_get_model:
            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(return_value=fake_response)
            mock_get_model.return_value = mock_model

            result = review_plan("Full trip plan here.")

        assert "Critique part 1" in result
        assert "Critique part 2" in result

    def test_uses_reviewer_prompt_from_registry(self):
        """get_prompt('reviewer_prompt') must be called — not a hardcoded string."""
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
