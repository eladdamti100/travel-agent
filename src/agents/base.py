"""
LLM factory — returns a configured model instance for the selected provider.

Set LLM_PROVIDER=groq or LLM_PROVIDER=gemini (default) in .env.
HuggingFace / torch silence vars are set once in run.py before any import.
"""

from src.config.settings import settings


def get_model(temperature: float = 0, bind_tools: list = None):
    """
    Factory — returns a configured LLM instance.
    Switch providers by setting LLM_PROVIDER=groq (or gemini) in .env.
    """
    provider = settings.llm_provider
    model_name = settings.resolved_model_name

    if provider == "groq":
        from langchain_groq import ChatGroq
        model = ChatGroq(model=model_name, temperature=temperature)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

    if bind_tools:
        return model.bind_tools(bind_tools)
    return model
