"""
LLM factory — returns a configured model instance for the selected provider.

Set LLM_PROVIDER=groq or LLM_PROVIDER=gemini (default) in .env.
"""

import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

_DEFAULTS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.1-8b-instant",  # smarter, parallel tool calls, better planning
}

MODEL_NAME = os.getenv("LLM_MODEL", _DEFAULTS.get(PROVIDER, "gemini-2.5-flash"))


def get_model(temperature: float = 0, bind_tools: list = None):
    """
    Factory — returns a configured LLM instance.
    Switch providers by setting LLM_PROVIDER=groq (or gemini) in .env.
    """
    if PROVIDER == "groq":
        from langchain_groq import ChatGroq
        model = ChatGroq(model=MODEL_NAME, temperature=temperature)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=temperature)

    if bind_tools:
        return model.bind_tools(bind_tools)
    return model
