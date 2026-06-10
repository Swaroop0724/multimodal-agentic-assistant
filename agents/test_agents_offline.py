"""
agents/test_agents_offline.py
Offline tests — validates agent structure and data types WITHOUT API calls.
Uses mocking to avoid real LLM/tool calls.
Run: python -m agents.test_agents_offline
"""

import sys
sys.path.insert(0, ".")

from unittest.mock import MagicMock, patch
from agents.base_agent import AgentResult


# ── Test 1: AgentResult dataclass ───────────────────────────────────────────────
def test_agent_result():
    print("\n── AgentResult Dataclass Tests ──────────────────────")
    passed = 0

    # Default values
    r = AgentResult(output="Hello")
    assert r.output == "Hello",            "output should be set"
    assert r.tool_calls == [],             "tool_calls should default to []"
    assert r.success is True,             "success should default to True"
    assert r.latency_ms == 0.0,           "latency_ms should default to 0.0"
    assert r.plan is None,                "plan should default to None"
    print("  ✅ Default values correct")
    passed += 1

    # Custom values
    r2 = AgentResult(
        output="Result",
        tool_calls=["web_search", "calculator"],
        tool_call_count=2,
        iterations=3,
        latency_ms=1234.5,
        success=True,
    )
    assert r2.tool_call_count == 2,       "tool_call_count should be 2"
    assert r2.latency_ms == 1234.5,       "latency_ms should be 1234.5"
    assert len(r2.tool_calls) == 2,       "should have 2 tool calls"
    print("  ✅ Custom values correct")
    passed += 1

    # Error result
    r3 = AgentResult(output="error", success=False, error="API timeout")
    assert r3.success is False,           "success should be False"
    assert r3.error == "API timeout",     "error message should be set"
    print("  ✅ Error result correct")
    passed += 1

    print(f"\n  {passed}/3 passed")
    return passed == 3


# ── Test 2: ReAct agent tool call extraction ────────────────────────────────────
def test_react_tool_extraction():
    print("\n── ReAct Tool Call Extraction Tests ─────────────────")

    # Mock AIMessage with tool_calls
    class MockToolCall:
        def __init__(self, name):
            self.name = name

    class MockAIMessage:
        __name__ = "AIMessage"
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []
            self.__class__.__name__ = "AIMessage"

    # Simulate what LangGraph returns
    messages = [
        MockAIMessage("I'll search for that", tool_calls=[{"name": "web_search"}]),
        MockAIMessage("Now I'll calculate", tool_calls=[{"name": "calculator"}]),
        MockAIMessage("The answer is 42"),
    ]

    # Extract tool calls (same logic as react_agent.py)
    tool_calls_made = []
    for msg in messages:
        if msg.__class__.__name__ == "AIMessage":
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.get("name") if isinstance(tc, dict) else tc.name
                    tool_calls_made.append(tool_name)

    assert tool_calls_made == ["web_search", "calculator"], \
        f"Expected ['web_search', 'calculator'], got {tool_calls_made}"
    print(f"  ✅ Extracted tools: {tool_calls_made}")

    # Final answer extraction
    final_answer = ""
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            final_answer = msg.content
            break
    assert final_answer == "The answer is 42"
    print(f"  ✅ Final answer: '{final_answer}'")

    print(f"\n  2/2 passed")
    return True


# ── Test 3: Plan-Execute plan parsing ───────────────────────────────────────────
def test_plan_parsing():
    print("\n── Plan-Execute Plan Parsing Tests ──────────────────")
    passed = 0

    def parse_plan(raw_plan: str) -> list[str]:
        """Same logic as plan_execute_agent.py planner_node."""
        steps = []
        for line in raw_plan.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                step = line.lstrip("0123456789.-) ").strip()
                if step:
                    steps.append(step)
        return steps if steps else [raw_plan]

    # Numbered list
    plan1 = "1. Search for GDP\n2. Calculate percentage\n3. Format the answer"
    result1 = parse_plan(plan1)
    assert result1 == ["Search for GDP", "Calculate percentage", "Format the answer"]
    print(f"  ✅ Numbered list: {result1}")
    passed += 1

    # Dashed list
    plan2 = "- Search web\n- Run calculation"
    result2 = parse_plan(plan2)
    assert result2 == ["Search web", "Run calculation"]
    print(f"  ✅ Dashed list: {result2}")
    passed += 1

    # Fallback: plain text
    plan3 = "Just answer the question directly"
    result3 = parse_plan(plan3)
    assert result3 == ["Just answer the question directly"]
    print(f"  ✅ Fallback (plain text): {result3}")
    passed += 1

    print(f"\n  {passed}/3 passed")
    return passed == 3


# ── Test 4: should_continue routing logic ───────────────────────────────────────
def test_routing_logic():
    print("\n── Plan-Execute Routing Logic Tests ─────────────────")
    passed = 0

    def should_continue(state: dict) -> str:
        completed = len(state["completed_steps"])
        total = len(state["plan"])
        if completed < total:
            return "executor"
        return "synthesizer"

    # More steps remaining
    s1 = {"plan": ["step1", "step2", "step3"], "completed_steps": ["r1"]}
    assert should_continue(s1) == "executor"
    print(f"  ✅ 1/3 done → 'executor'")
    passed += 1

    # All steps done
    s2 = {"plan": ["step1", "step2"], "completed_steps": ["r1", "r2"]}
    assert should_continue(s2) == "synthesizer"
    print(f"  ✅ 2/2 done → 'synthesizer'")
    passed += 1

    # Empty plan edge case
    s3 = {"plan": [], "completed_steps": []}
    assert should_continue(s3) == "synthesizer"
    print(f"  ✅ Empty plan → 'synthesizer'")
    passed += 1

    print(f"\n  {passed}/3 passed")
    return passed == 3


if __name__ == "__main__":
    print("=" * 55)
    print("  Multimodal Agentic Assistant — Agent Tests")
    print("=" * 55)

    results = [
        test_agent_result(),
        test_react_tool_extraction(),
        test_plan_parsing(),
        test_routing_logic(),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*55}")
    print(f"  Overall: {passed}/{total} test suites passed")
    print(f"{'='*55}")
    sys.exit(0 if passed == total else 1)
