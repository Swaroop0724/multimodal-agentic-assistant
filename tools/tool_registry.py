"""
tools/tool_registry.py
Central registry — import get_tools() anywhere to get all 5 tools.
Also provides tool lookup by name for benchmark logging.
"""

from tools.web_search_tool import web_search
from tools.code_executor_tool import code_executor
from tools.vision_tool import vision_analyzer
from tools.calculator_tool import calculator
from tools.wikipedia_tool import wikipedia_search
from loguru import logger


def get_tools() -> list:
    """
    Return all 5 agent tools as a list.
    Pass directly to LangChain/LangGraph agent constructors.
    """
    tools = [
        web_search,
        code_executor,
        vision_analyzer,
        calculator,
        wikipedia_search,
    ]
    logger.debug(f"Tool registry loaded: {[t.name for t in tools]}")
    return tools


def get_tool_by_name(name: str):
    """Look up a single tool by its .name attribute."""
    tools = get_tools()
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool '{name}' not found. Available: {[t.name for t in tools]}")


# ── Tool metadata for UI display ────────────────────────────────────────────────
TOOL_METADATA = {
    "web_search": {
        "icon": "🌐",
        "description": "Real-time web search via Tavily",
        "use_case": "Current events, prices, recent news",
        "provider": "Tavily",
    },
    "code_executor": {
        "icon": "💻",
        "description": "Sandboxed Python execution via E2B",
        "use_case": "Algorithms, data processing, computation",
        "provider": "E2B",
    },
    "vision_analyzer": {
        "icon": "👁️",
        "description": "Image analysis via Groq LLaMA Vision",
        "use_case": "Image understanding, visual Q&A",
        "provider": "Groq",
    },
    "calculator": {
        "icon": "🔢",
        "description": "Safe math expression evaluator",
        "use_case": "Arithmetic, statistics, formulas",
        "provider": "Built-in",
    },
    "wikipedia_search": {
        "icon": "📚",
        "description": "Encyclopedic knowledge via Wikipedia",
        "use_case": "Concepts, history, science, definitions",
        "provider": "Wikipedia",
    },
}
