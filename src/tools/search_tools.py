"""
Web search tool — real-time travel information via Tavily.
Requires TAVILY_API_KEY in .env. Falls back gracefully when key is absent.
"""

import os

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time travel information not available in the local database.
    Use for current events, travel advisories, currency rates, or any up-to-date info.
    query: natural-language search string.
    Requires TAVILY_API_KEY to be set in .env.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return (
            "Web search is unavailable: TAVILY_API_KEY is not set. "
            "Add it to your .env file to enable real-time search."
        )

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults

        search = TavilySearchResults(max_results=3, tavily_api_key=api_key)
        results = search.invoke(query)
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"Source: {r.get('url', 'N/A')}\n{r.get('content', '')}" for r in results
        )
    except ImportError:
        return (
            "Web search requires langchain-community. "
            "Install it with: pip install langchain-community"
        )
    except Exception as e:
        return f"Web search error: {e}"
