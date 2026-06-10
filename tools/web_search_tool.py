"""
tools/web_search_tool.py
Tavily-powered web search tool for real-time information retrieval.
Returns top 3 results with title, url, and content snippet.
"""

from langchain.tools import tool
from tavily import TavilyClient
from config.settings import settings
from loguru import logger


@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time information.
    Use this for: current events, latest news, prices, recent releases,
    anything that requires up-to-date information beyond training data.

    Args:
        query: The search query string

    Returns:
        Formatted string with top search results
    """
    try:
        logger.debug(f"[web_search] query='{query}'")
        client = TavilyClient(api_key=settings.tavily_api_key)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=3,
            include_answer=True,        # Tavily's AI summary
            include_raw_content=False,
        )

        # ── Format results ──────────────────────────────────────────────────────
        output_parts = []

        # AI-generated direct answer (if available)
        if response.get("answer"):
            output_parts.append(f"Direct Answer: {response['answer']}\n")

        # Individual search results
        results = response.get("results", [])
        if results:
            output_parts.append("Search Results:")
            for i, r in enumerate(results, 1):
                output_parts.append(
                    f"\n[{i}] {r.get('title', 'No title')}\n"
                    f"    URL: {r.get('url', '')}\n"
                    f"    {r.get('content', '')[:300]}..."
                )

        if not output_parts:
            return f"No results found for query: {query}"

        result = "\n".join(output_parts)
        logger.debug(f"[web_search] returned {len(results)} results")
        return result

    except Exception as e:
        logger.error(f"[web_search] error: {e}")
        return f"Web search failed: {str(e)}"
