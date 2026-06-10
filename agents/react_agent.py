"""
agents/react_agent.py

ReAct Agent — Reasoning + Acting in an interleaved loop.

Loop:
    Thought  →  "I need to search for X"
    Action   →  web_search("X")
    Observe  →  "Result: ..."
    Thought  →  "Now I can answer..."
    Final    →  "The answer is..."

Built with LangGraph's prebuilt create_react_agent.
"""

from typing import Optional
from loguru import logger

from agents.base_agent import BaseAgent, AgentResult
from config.settings import settings


REACT_SYSTEM_PROMPT = """You are a powerful multimodal AI assistant with access to 5 tools:

1. web_search       — Search the web for real-time / current information
2. code_executor    — Write and execute Python code in a secure sandbox
3. vision_analyzer  — Analyze images (provide image URL or file path + your question)
4. calculator       — Evaluate mathematical expressions safely
5. wikipedia_search — Look up factual / encyclopedic knowledge

Guidelines:
- Always choose the MOST appropriate tool for the query
- For math problems: use calculator (not code_executor unless the user asks for code)
- For current events / prices: use web_search (not wikipedia)
- For concepts / history / science: use wikipedia_search
- For coding tasks: use code_executor
- For image questions: use vision_analyzer with the image URL/path
- You may call multiple tools in sequence if needed
- Always cite which tools you used at the end

CRITICAL RULES FOR CODE:
- When the user asks for code, ALWAYS show the full code in your final answer inside a markdown code block like ```python ... ```
- ALWAYS show the actual output/result from running the code
- Never just say "I used code_executor" — always paste the actual code AND its output
- Format: explanation → ```python code here ``` → "Output: ..." → brief summary

CRITICAL RULES FOR ALL TOOLS:
- Always include the actual content returned by tools in your answer
- Never summarize without showing the real data/code/result
- Be complete — users want to SEE the actual output, not just be told it exists
"""


class ReActAgent(BaseAgent):
    """
    ReAct agent using LangGraph's create_react_agent.
    Interleaves reasoning and tool calls in a tight loop.
    Terminates when the LLM outputs a final answer without a tool call.
    """

    def __init__(self, model_name: str = "llama-70b"):
        super().__init__(model_name)
        logger.info(f"ReActAgent initialized | model={self.model_id}")

    def _build_graph(self):
        """Build LangGraph ReAct agent graph."""
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import SystemMessage

        graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SystemMessage(content=REACT_SYSTEM_PROMPT),
        )
        return graph

    def run(self, query: str, image_path: Optional[str] = None) -> AgentResult:
        """
        Run the ReAct agent on a query.

        Args:
            query: User query
            image_path: Optional image URL/path — appended to query for vision tasks

        Returns:
            AgentResult with output, tool usage, latency
        """
        from langchain_core.messages import HumanMessage

        # ── Build input message ────────────────────────────────────────────────
        if image_path:
            # Multimodal: include image reference in query
            full_query = f"{query}\n\nImage to analyze: {image_path}"
        else:
            full_query = query

        input_messages = {"messages": [HumanMessage(content=full_query)]}

        logger.info(f"[ReAct] Running query: '{query[:80]}...' | model={self.model_id}")

        # ── Execute with timing ────────────────────────────────────────────────
        try:
            def _invoke():
                return self.graph.invoke(
                    input_messages,
                    config={"recursion_limit": settings.max_agent_iterations * 2}
                )

            raw_result, latency_ms = self._timed_run(_invoke)

            # ── Parse output ───────────────────────────────────────────────────
            messages = raw_result.get("messages", [])

            # Final answer is the last AI message
            final_answer = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    # Skip tool messages
                    if msg.__class__.__name__ in ("AIMessage", "HumanMessage"):
                        if msg.__class__.__name__ == "AIMessage":
                            final_answer = msg.content
                            break

            # ── Extract tool calls from message history ────────────────────────
            tool_calls_made = []
            iterations = 0
            for msg in messages:
                cls = msg.__class__.__name__
                if cls == "AIMessage":
                    iterations += 1
                    # LangGraph AIMessage has tool_calls attribute
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name") if isinstance(tc, dict) else tc.name
                            tool_calls_made.append(tool_name)

            logger.info(
                f"[ReAct] Done | latency={latency_ms:.0f}ms | "
                f"tools={tool_calls_made} | iterations={iterations}"
            )

            return AgentResult(
                output=final_answer or "No answer generated.",
                tool_calls=tool_calls_made,
                tool_call_count=len(tool_calls_made),
                iterations=iterations,
                latency_ms=latency_ms,
                success=True,
                intermediate_steps=messages,
            )

        except Exception as e:
            logger.error(f"[ReAct] Error: {e}")
            return AgentResult(
                output=f"Agent error: {str(e)}",
                tool_calls=[],
                tool_call_count=0,
                iterations=0,
                latency_ms=0.0,
                success=False,
                error=str(e),
            )