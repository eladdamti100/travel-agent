import importlib
from typing import Any

PROMPT_REGISTRY = {
    "planner_prompt": "src.prompts.planner_prompts.PLANNER_SYSTEM_PROMPT",
    "final_answer_prompt": "src.prompts.final_answer_prompts.FINAL_ANSWER_PROMPT",  # מופרד לקובץ משלו!
    "validator_prompt": "src.prompts.validator_prompts.VALIDATOR_SYSTEM_PROMPT",
    "reviewer_prompt": "src.prompts.reviewer_prompts.REVIEWER_SYSTEM_PROMPT",
    "orchestrator_prompt": "src.prompts.orchestrator_prompts.ORCHESTRATOR_SYSTEM_PROMPT",
    "researcher_prompt": "src.prompts.researcher_prompts.RESEARCHER_SYSTEM_PROMPT",
}

def get_prompt(prompt_name: str) -> str:
    if prompt_name not in PROMPT_REGISTRY:
        raise ValueError(f"Prompt '{prompt_name}' not found in registry.")
    
    full_path = PROMPT_REGISTRY[prompt_name]
    module_path, attr_name = full_path.rsplit(".", 1)
    
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load prompt '{prompt_name}' from {full_path}: {e}")