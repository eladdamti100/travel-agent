"""
Tests for cache compression service — compress_answer
"""

from unittest.mock import patch, MagicMock


class TestCompressAnswer:

    def test_short_answer_returned_without_llm_call(self):
        from src.services.cache_compression import compress_answer, _MIN_COMPRESS_CHARS
        short = "x" * (_MIN_COMPRESS_CHARS - 1)
        with patch("src.agents.base.get_model") as mock_get_model:
            result = compress_answer(short)
        mock_get_model.assert_not_called()
        assert result == short

    def test_long_answer_compressed_via_llm(self):
        from src.services.cache_compression import compress_answer, _MIN_COMPRESS_CHARS
        long_answer = "A" * (_MIN_COMPRESS_CHARS + 100)
        fake_response = MagicMock()
        fake_response.content = "• Bullet 1\n• Bullet 2"
        mock_model = MagicMock()
        mock_model.invoke.return_value = fake_response
        with patch("src.agents.base.get_model", return_value=mock_model):
            result = compress_answer(long_answer)
        assert result == "• Bullet 1\n• Bullet 2"
        mock_model.invoke.assert_called_once()

    def test_llm_failure_falls_back_to_truncation(self):
        from src.services.cache_compression import compress_answer, _MIN_COMPRESS_CHARS
        long_answer = "B" * 1000
        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("LLM down")
        with patch("src.agents.base.get_model", return_value=mock_model):
            result = compress_answer(long_answer)
        assert result.endswith("...")
        assert len(result) <= 503
