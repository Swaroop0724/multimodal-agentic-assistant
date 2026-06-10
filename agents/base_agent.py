"""
agents/base_agent.py
Abstract base class that both ReActAgent and PlanExecuteAgent inherit from.
Defines the common interface used by benchmark harness and UI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class AgentResult:
    """
    Standardized output from any agent run.
    Both ReAct and Plan-Execute return this exact structure
    so the benchmark harness can compare them apples-to-apples.
    """
    output: str                          # Final answer text
    tool_calls: list[str] = field(default_factory=list)   # Names of tools invoked
    tool_call_count: int = 0             # Total number of tool calls made
    iterations: int = 0                  # Number of agent loop iterations
    latency_ms: float = 0.0             # Total wall-clock time in milliseconds
    success: bool = True                 # False if agent errored or timed out
    error: Optional[str] = None          # Error message if success=False
    intermediate_steps: list = field(default_factory=list)  # Raw LangGraph steps
    plan: Optional[list[str]] = None     # Only populated by PlanExecuteAgent


class BaseAgent(ABC):
    """
    Abstract base for ReAct and Plan-Execute agents.
    Subclasses must implement run() and _build_graph().
    """

    def __init__(self, model_name: str):
        from config.settings import settings, MODEL_REGISTRY
        from langchain_groq import ChatGroq
        from tools.tool_registry import get_tools

        self.model_name = model_name
        self.model_id = MODEL_REGISTRY.get(model_name, model_name)
        self.tools = get_tools()
        self.tool_map = {t.name: t for t in self.tools}

        # Shared LLM instance
        self.llm = ChatGroq(
            model=self.model_id,
            api_key=settings.groq_api_key,
            temperature=0,               # Deterministic for benchmarking
            max_tokens=4096,
        )

        self.graph = self._build_graph()

    @abstractmethod
    def _build_graph(self):
        """Build and compile the LangGraph state graph."""
        pass

    @abstractmethod
    def run(self, query: str, image_path: Optional[str] = None) -> AgentResult:
        """
        Execute the agent on a query and return standardized AgentResult.

        Args:
            query: User's natural language query
            image_path: Optional path/URL to image for vision tasks

        Returns:
            AgentResult with output, tool usage, latency
        """
        pass

    def _timed_run(self, fn) -> tuple:
        """Utility: run fn() and return (result, elapsed_ms)."""
        start = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def _extract_tool_calls(self, steps: list) -> list[str]:
        """Extract tool names from LangGraph intermediate steps."""
        tool_calls = []
        for step in steps:
            # LangGraph returns (AgentAction, observation) tuples
            if isinstance(step, tuple) and len(step) == 2:
                action = step[0]
                if hasattr(action, "tool"):
                    tool_calls.append(action.tool)
            # Or direct AgentAction objects
            elif hasattr(step, "tool"):
                tool_calls.append(step.tool)
        return tool_calls
