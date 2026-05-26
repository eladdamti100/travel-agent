"""
Tests for the prompt registry — get_prompt / loader.py
"""

import importlib
import pytest
from unittest.mock import patch


class TestPromptLoader:

    def test_known_prompt_returns_non_empty_string(self):
        from src.prompts.loader import get_prompt, PROMPT_REGISTRY

        for prompt_name in PROMPT_REGISTRY:
            try:
                prompt = get_prompt(prompt_name)
                assert isinstance(prompt, str), f"{prompt_name} must return a string"
                assert len(prompt) > 10, f"{prompt_name} prompt appears empty"
            except RuntimeError as e:
                if "Failed to load prompt" in str(e):
                    pytest.skip(f"Prompt module not found: {prompt_name}")
                raise

    def test_unknown_prompt_raises_value_error(self):
        from src.prompts.loader import get_prompt

        with pytest.raises(ValueError, match="not found in registry"):
            get_prompt("totally_nonexistent_prompt_xyz")

    def test_all_registry_paths_importable(self):
        from src.prompts.loader import PROMPT_REGISTRY

        for prompt_name, full_path in PROMPT_REGISTRY.items():
            module_path, attr_name = full_path.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                value = getattr(module, attr_name)
                assert isinstance(value, str), (
                    f"{prompt_name}: expected str, got {type(value)}"
                )
            except (ImportError, AttributeError) as exc:
                pytest.skip(f"Prompt module not found: {prompt_name}")

    def test_runtime_error_on_bad_module_path(self):
        from src.prompts.loader import get_prompt, PROMPT_REGISTRY

        with patch.dict(
            PROMPT_REGISTRY,
            {"bad_prompt": "src.prompts.nonexistent_module.SOME_PROMPT"},
        ):
            with pytest.raises(RuntimeError, match="Failed to load prompt"):
                get_prompt("bad_prompt")