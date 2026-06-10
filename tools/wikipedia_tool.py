"""
tools/wikipedia_tool.py
Wikipedia-powered knowledge search for factual, encyclopedic information.
Returns a clean summary — no HTML, no citations markup.
"""

from langchain.tools import tool
from loguru import logger
import wikipedia
import re


def _clean_text(text: str) -> str:
    """Remove citation markers like [1], [2] and extra whitespace."""
    text = re.sub(r"\[\d+\]", "", text)       # Remove [1], [23] etc.
    text = re.sub(r"\s+", " ", text)           # Collapse whitespace
    return text.strip()


@tool
def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia for factual, encyclopedic information.
    Use this for: concepts, definitions, historical facts, scientific principles,
    biographies, technical explanations, or any well-established knowledge.
    Prefer web_search for current events or recent information.

    Args:
        query: Topic or concept to look up

    Returns:
        Wikipedia summary (up to 1500 chars) with page title and URL
    """
    try:
        logger.debug(f"[wikipedia_search] query='{query}'")

        # Configure language
        wikipedia.set_lang("en")

        # ── Search for best matching page ───────────────────────────────────────
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return f"No Wikipedia results found for: {query}"

        # Try pages in order until one loads without disambiguation error
        page = None
        page_title = None
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                page_title = title
                break
            except wikipedia.exceptions.DisambiguationError as e:
                # Take first option from disambiguation
                try:
                    page = wikipedia.page(e.options[0], auto_suggest=False)
                    page_title = e.options[0]
                    break
                except Exception:
                    continue
            except wikipedia.exceptions.PageError:
                continue

        if page is None:
            return f"Could not load Wikipedia page for: {query}"

        # ── Extract and format content ──────────────────────────────────────────
        summary = _clean_text(wikipedia.summary(page_title, sentences=5))

        # Also get first 800 chars of full content for more detail
        full_content = _clean_text(page.content)
        detail = full_content[:800] if len(full_content) > len(summary) else ""

        output = f"Wikipedia: {page.title}\nURL: {page.url}\n\n{summary}"
        if detail and detail != summary[:len(detail)]:
            output += f"\n\nAdditional Detail:\n{detail}..."

        logger.debug(f"[wikipedia_search] found page='{page.title}' len={len(output)}")
        return output

    except Exception as e:
        logger.error(f"[wikipedia_search] error: {e}")
        return f"Wikipedia search failed: {str(e)}"
