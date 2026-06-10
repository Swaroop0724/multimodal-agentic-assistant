"""
tools/test_tools_offline.py
Offline tests — validates tool logic WITHOUT API calls.
Run: python -m tools.test_tools_offline
"""

import sys
sys.path.insert(0, ".")

# ── Test 1: Calculator (no API needed) ──────────────────────────────────────────
def test_calculator():
    print("\n── Calculator Tests ──────────────────────────────")
    from tools.calculator_tool import _evaluate_expression

    tests = [
        ("2 + 2",                        4),
        ("10000 * (1 + 0.07) ** 10",     19671.513571),
        ("sqrt(9**2 + 12**2)",           15.0),
        ("(98.6 - 32) * 5/9",           37.0),
        ("log2(1024)",                   10.0),
        ("round(pi, 4)",                 3.1416),
    ]

    passed = 0
    for expr, expected in tests:
        result = _evaluate_expression(expr)
        ok = abs(result - expected) < 0.01
        status = "✅" if ok else "❌"
        print(f"  {status} {expr} = {result:.4f} (expected ≈ {expected})")
        if ok:
            passed += 1

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


# ── Test 2: Code extraction from markdown ────────────────────────────────────────
def test_code_extraction():
    print("\n── Code Extraction Tests ────────────────────────────")
    from tools.code_executor_tool import _extract_code

    tests = [
        ("print('hello')",               "print('hello')"),
        ("```python\nprint('hi')\n```",  "print('hi')"),
        ("```\nx = 1 + 1\n```",          "x = 1 + 1"),
    ]

    passed = 0
    for input_text, expected in tests:
        result = _extract_code(input_text)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} Input: {repr(input_text[:30])} → {repr(result)}")
        if ok:
            passed += 1

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


# ── Test 3: Wikipedia text cleaning ─────────────────────────────────────────────
def test_wiki_cleaning():
    print("\n── Wikipedia Text Cleaning Tests ───────────────────")
    from tools.wikipedia_tool import _clean_text

    tests = [
        ("Albert Einstein[1] was born[23]", "Albert Einstein was born"),
        ("Hello   world\n\ntest",            "Hello world test"),
    ]

    passed = 0
    for input_text, expected in tests:
        result = _clean_text(input_text)
        ok = result == expected
        status = "✅" if ok else "❌"
        print(f"  {status} '{input_text}' → '{result}'")
        if ok:
            passed += 1

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


# ── Test 4: Tool registry loads correctly ────────────────────────────────────────
def test_tool_registry():
    print("\n── Tool Registry Test ───────────────────────────────")
    try:
        from tools.tool_registry import get_tools, TOOL_METADATA

        tools = get_tools()
        tool_names = [t.name for t in tools]

        expected = ["web_search", "code_executor", "vision_analyzer", "calculator", "wikipedia_search"]
        ok = set(tool_names) == set(expected)

        print(f"  {'✅' if ok else '❌'} Tools loaded: {tool_names}")
        print(f"  {'✅' if len(TOOL_METADATA) == 5 else '❌'} Metadata entries: {len(TOOL_METADATA)}")

        # Verify each tool has required attributes
        for tool in tools:
            has_name = hasattr(tool, "name")
            has_desc = hasattr(tool, "description")
            print(f"    {'✅' if has_name and has_desc else '❌'} {tool.name}: name={has_name}, description={has_desc}")

        return ok
    except Exception as e:
        print(f"  ❌ Registry failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 55)
    print("  Multimodal Agentic Assistant — Offline Tool Tests")
    print("=" * 55)

    results = [
        test_calculator(),
        test_code_extraction(),
        test_wiki_cleaning(),
        test_tool_registry(),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*55}")
    print(f"  Overall: {passed}/{total} test suites passed")
    print(f"{'='*55}")

    sys.exit(0 if passed == total else 1)
