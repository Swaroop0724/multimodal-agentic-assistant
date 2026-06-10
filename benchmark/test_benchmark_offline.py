"""
benchmark/test_benchmark_offline.py
Tests all benchmark metric calculations without any API calls.
Run: python -m benchmark.test_benchmark_offline
"""

import sys
import csv
import json
import tempfile
from pathlib import Path
sys.path.insert(0, ".")


# ── Test 1: ROUGE-L computation ─────────────────────────────────────────────────
def test_rouge_l():
    print("\n── ROUGE-L Tests ─────────────────────────────────────")
    from benchmark.metrics import compute_rouge_l
    passed = 0

    tests = [
        # (hypothesis, reference, min_expected, max_expected)
        ("quantum computing breakthroughs error correction", "quantum computing breakthroughs error correction", 0.9, 1.0),
        ("the cat sat on the mat", "the dog sat on the mat", 0.5, 0.9),
        ("completely unrelated answer here", "quantum computing machine learning", 0.0, 0.3),
        ("", "reference answer", 0.0, 0.0),
        ("some answer", "", 0.0, 0.0),
    ]

    for hyp, ref, min_e, max_e in tests:
        score = compute_rouge_l(hyp, ref)
        ok = min_e <= score <= max_e
        print(f"  {'✅' if ok else '❌'} ROUGE-L={score:.4f} (expected [{min_e}, {max_e}])")
        print(f"       hyp: '{hyp[:40]}'")
        if ok:
            passed += 1

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


# ── Test 2: Tool accuracy ────────────────────────────────────────────────────────
def test_tool_accuracy():
    print("\n── Tool Accuracy Tests ────────────────────────────────")
    from benchmark.metrics import compute_tool_accuracy
    passed = 0

    tests = [
        # (tool_calls_made, expected_tool, expected_score)
        (["web_search"],            "web_search",              1.0),
        (["calculator"],            "calculator",              1.0),
        ([],                        "web_search",              0.0),
        (["calculator"],            "web_search",              0.0),
        (["web_search", "calculator"], "web_search+calculator", 1.0),   # multi-tool
        (["web_search"],            "web_search+calculator",   0.5),    # partial credit
        (["web_search"],            "",                        1.0),    # no expectation
    ]

    for calls, expected, exp_score in tests:
        score = compute_tool_accuracy(calls, expected)
        ok = abs(score - exp_score) < 0.01
        print(f"  {'✅' if ok else '❌'} calls={calls} expected='{expected}' → {score:.2f} (want {exp_score})")
        if ok:
            passed += 1

    print(f"\n  {passed}/{len(tests)} passed")
    return passed == len(tests)


# ── Test 3: Aggregate metrics ────────────────────────────────────────────────────
def test_aggregate_metrics():
    print("\n── Aggregate Metrics Tests ───────────────────────────")
    from benchmark.metrics import QueryMetrics, aggregate_metrics
    passed = 0

    # Create 4 mock QueryMetrics
    metrics = [
        QueryMetrics("Q1", "web_search", "react", "llama-70b",
                     "q1", "web_search", "ref1", "out1",
                     ["web_search"], 1200, True, 0.7, 1.0),
        QueryMetrics("Q2", "calculator", "react", "llama-70b",
                     "q2", "calculator", "ref2", "out2",
                     ["calculator"], 800, True, 0.65, 1.0),
        QueryMetrics("Q3", "code_execution", "react", "llama-70b",
                     "q3", "code_executor", "ref3", "out3",
                     ["code_executor"], 2000, True, 0.8, 1.0),
        QueryMetrics("Q4", "web_search", "react", "llama-70b",
                     "q4", "web_search", "ref4", "out4",
                     [], 500, False, 0.0, 0.0, "timeout"),
    ]

    agg = aggregate_metrics(metrics)

    # 3 successful, 1 failed
    assert agg["total_queries"] == 4,          f"total={agg['total_queries']}"
    assert agg["successful_queries"] == 3,     f"successful={agg['successful_queries']}"
    assert agg["failed_queries"] == 1,         f"failed={agg['failed_queries']}"
    assert agg["success_rate"] == 0.75,        f"success_rate={agg['success_rate']}"
    print(f"  ✅ Counts: total=4, success=3, fail=1, rate=75%")
    passed += 1

    # ROUGE-L average of successful: (0.7 + 0.65 + 0.8) / 3 = 0.7167
    expected_rouge = round((0.7 + 0.65 + 0.8) / 3, 4)
    assert abs(agg["avg_rouge_l"] - expected_rouge) < 0.001, \
        f"avg_rouge_l={agg['avg_rouge_l']} expected={expected_rouge}"
    print(f"  ✅ ROUGE-L avg={agg['avg_rouge_l']:.4f} (expected {expected_rouge:.4f})")
    passed += 1

    # Tool accuracy average of successful: (1.0 + 1.0 + 1.0) / 3 = 1.0
    assert agg["avg_tool_accuracy"] == 1.0,    f"tool_acc={agg['avg_tool_accuracy']}"
    print(f"  ✅ Tool accuracy={agg['avg_tool_accuracy']}")
    passed += 1

    # Latency average of successful: (1200 + 800 + 2000) / 3 = 1333.3
    expected_lat = round((1200 + 800 + 2000) / 3, 1)
    assert abs(agg["avg_latency_ms"] - expected_lat) < 1.0, \
        f"latency={agg['avg_latency_ms']} expected={expected_lat}"
    print(f"  ✅ Avg latency={agg['avg_latency_ms']:.1f}ms (expected {expected_lat:.1f}ms)")
    passed += 1

    # Category breakdown exists
    assert "web_search" in agg["by_category"],  "web_search category missing"
    assert "calculator" in agg["by_category"],  "calculator category missing"
    print(f"  ✅ Category breakdown: {list(agg['by_category'].keys())}")
    passed += 1

    print(f"\n  {passed}/5 passed")
    return passed == 5


# ── Test 4: Agent comparison ─────────────────────────────────────────────────────
def test_compare_agents():
    print("\n── Agent Comparison Tests ────────────────────────────")
    from benchmark.metrics import QueryMetrics, compare_agents
    passed = 0

    def make_metrics(agent_type, rouge, tool_acc, latency, success=True):
        return QueryMetrics(
            "Q1", "web_search", agent_type, "llama-70b",
            "query", "web_search", "reference", "output",
            ["web_search"] if tool_acc == 1.0 else [],
            latency, success, rouge, tool_acc
        )

    # ReAct: better ROUGE, better accuracy, lower latency
    react_metrics = [make_metrics("react", 0.71, 0.873, 2300)]
    pe_metrics    = [make_metrics("plan_execute", 0.63, 0.712, 3500)]

    comp = compare_agents(react_metrics, pe_metrics)

    assert comp["comparison"]["rouge_l_winner"] == "react",         "react should win ROUGE-L"
    assert comp["comparison"]["latency_winner"] == "react",         "react should win latency"
    assert comp["comparison"]["tool_accuracy_winner"] == "react",   "react should win accuracy"
    print(f"  ✅ ReAct wins: ROUGE-L, latency, tool accuracy")
    passed += 1

    lat_adv = comp["comparison"]["react_latency_advantage_pct"]
    assert lat_adv > 0, f"latency advantage should be > 0, got {lat_adv}"
    print(f"  ✅ ReAct latency advantage: {lat_adv:.1f}%")
    passed += 1

    print(f"\n  {passed}/2 passed")
    return passed == 2


# ── Test 5: CSV reporter ─────────────────────────────────────────────────────────
def test_reporter():
    print("\n── Reporter Tests ────────────────────────────────────")
    from benchmark.reporter import load_results_from_csv, generate_text_report

    # Create a temp CSV with mock data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "category", "agent_type", "model_name",
            "query", "expected_tool", "tool_calls_made",
            "rouge_l", "tool_accuracy", "latency_ms",
            "success", "error", "actual_output"
        ])
        writer.writerow(["Q1", "web_search", "react", "llama-70b",
                         "test query", "web_search", "web_search",
                         "0.71", "1.0", "2300", "True", "", "answer"])
        writer.writerow(["Q1", "web_search", "plan_execute", "llama-70b",
                         "test query", "web_search", "web_search",
                         "0.63", "1.0", "3500", "True", "", "answer"])
        tmp_path = f.name

    react_rows, pe_rows = load_results_from_csv(tmp_path)
    assert len(react_rows) == 1,  f"Expected 1 react row, got {len(react_rows)}"
    assert len(pe_rows) == 1,     f"Expected 1 pe row, got {len(pe_rows)}"
    print(f"  ✅ CSV loaded: {len(react_rows)} react, {len(pe_rows)} plan-execute rows")

    report = generate_text_report(tmp_path)
    assert "BENCHMARK REPORT" in report,    "Report should have header"
    assert "ReAct" in report,               "Report should mention ReAct"
    assert "RESUME BULLET" in report,       "Report should have resume bullet"
    print(f"  ✅ Text report generated ({len(report)} chars)")
    print(f"  ✅ Contains: header, ReAct, resume bullet")

    Path(tmp_path).unlink()

    print(f"\n  2/2 passed")
    return True


if __name__ == "__main__":
    print("=" * 55)
    print("  Benchmark — Offline Tests")
    print("=" * 55)

    results = [
        test_rouge_l(),
        test_tool_accuracy(),
        test_aggregate_metrics(),
        test_compare_agents(),
        test_reporter(),
    ]

    total = len(results)
    passed = sum(results)
    print(f"\n{'='*55}")
    print(f"  Overall: {passed}/{total} test suites passed")
    print(f"{'='*55}")
    sys.exit(0 if passed == total else 1)
