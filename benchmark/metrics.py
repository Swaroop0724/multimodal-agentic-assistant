"""
benchmark/metrics.py
All metric calculations for the benchmark.

Metrics computed:
  - ROUGE-L        : longest common subsequence overlap with reference answer
  - Tool Accuracy  : did the agent call the expected tool?
  - Latency (ms)   : wall-clock time per query
  - Success Rate   : % of queries that completed without error
  - Tool Call Rate : avg number of tool calls per query
"""

from rouge_score import rouge_scorer
from dataclasses import dataclass
from typing import Optional
import statistics


@dataclass
class QueryMetrics:
    """Metrics for a single query run."""
    query_id: str
    category: str
    agent_type: str          # "react" or "plan_execute"
    model_name: str
    query: str
    expected_tool: str
    reference_answer: str
    actual_output: str
    tool_calls_made: list[str]
    latency_ms: float
    success: bool
    rouge_l: float = 0.0
    tool_accuracy: float = 0.0
    error: Optional[str] = None


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L score between hypothesis and reference.
    ROUGE-L measures the longest common subsequence — captures
    sentence-level structure similarity.

    Returns float in [0.0, 1.0]
    """
    if not hypothesis or not reference:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference.lower(), hypothesis.lower())
    return round(scores["rougeL"].fmeasure, 4)


def compute_tool_accuracy(
    tool_calls_made: list[str],
    expected_tool: str,
) -> float:
    """
    Check if the agent called the expected tool.

    For multi-tool queries (expected_tool contains '+'),
    partial credit is given for each correct tool called.

    Returns float in [0.0, 1.0]
    """
    if not expected_tool:
        return 1.0  # No expectation = automatic pass

    expected_tools = [t.strip() for t in expected_tool.split("+")]

    if not tool_calls_made:
        return 0.0

    # Count how many expected tools were actually called
    matched = sum(
        1 for exp in expected_tools
        if any(exp in called for called in tool_calls_made)
    )

    return round(matched / len(expected_tools), 4)


def aggregate_metrics(query_metrics: list[QueryMetrics]) -> dict:
    """
    Aggregate per-query metrics into summary statistics.

    Returns dict with all benchmark summary numbers.
    """
    if not query_metrics:
        return {}

    successful = [m for m in query_metrics if m.success]
    failed = [m for m in query_metrics if not m.success]

    # ── Core metrics ────────────────────────────────────────────────────────────
    rouge_scores = [m.rouge_l for m in successful]
    latencies = [m.latency_ms for m in successful]
    tool_accuracies = [m.tool_accuracy for m in successful]
    tool_call_counts = [len(m.tool_calls_made) for m in successful]

    # ── Per-category breakdown ───────────────────────────────────────────────────
    categories = {}
    for m in query_metrics:
        cat = m.category
        if cat not in categories:
            categories[cat] = {"rouge_l": [], "tool_accuracy": [], "latency_ms": []}
        if m.success:
            categories[cat]["rouge_l"].append(m.rouge_l)
            categories[cat]["tool_accuracy"].append(m.tool_accuracy)
            categories[cat]["latency_ms"].append(m.latency_ms)

    category_summary = {}
    for cat, vals in categories.items():
        category_summary[cat] = {
            "avg_rouge_l": round(statistics.mean(vals["rouge_l"]), 4) if vals["rouge_l"] else 0,
            "avg_tool_accuracy": round(statistics.mean(vals["tool_accuracy"]), 4) if vals["tool_accuracy"] else 0,
            "avg_latency_ms": round(statistics.mean(vals["latency_ms"]), 1) if vals["latency_ms"] else 0,
            "count": len(vals["rouge_l"]),
        }

    return {
        # Overall
        "total_queries": len(query_metrics),
        "successful_queries": len(successful),
        "failed_queries": len(failed),
        "success_rate": round(len(successful) / len(query_metrics), 4),

        # ROUGE-L
        "avg_rouge_l": round(statistics.mean(rouge_scores), 4) if rouge_scores else 0,
        "median_rouge_l": round(statistics.median(rouge_scores), 4) if rouge_scores else 0,
        "min_rouge_l": round(min(rouge_scores), 4) if rouge_scores else 0,
        "max_rouge_l": round(max(rouge_scores), 4) if rouge_scores else 0,

        # Tool accuracy
        "avg_tool_accuracy": round(statistics.mean(tool_accuracies), 4) if tool_accuracies else 0,

        # Latency
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1) if latencies else 0,

        # Tool usage
        "avg_tool_calls_per_query": round(statistics.mean(tool_call_counts), 2) if tool_call_counts else 0,

        # Per-category breakdown
        "by_category": category_summary,
    }


def compare_agents(
    react_metrics: list[QueryMetrics],
    plan_execute_metrics: list[QueryMetrics],
) -> dict:
    """
    Side-by-side comparison of ReAct vs Plan-Execute.
    Returns the comparison dict used in the final report.
    """
    react_agg = aggregate_metrics(react_metrics)
    pe_agg = aggregate_metrics(plan_execute_metrics)

    def pct_diff(a, b):
        """Percentage difference: how much better is a vs b?"""
        if b == 0:
            return 0.0
        return round((a - b) / b * 100, 1)

    return {
        "react": react_agg,
        "plan_execute": pe_agg,
        "comparison": {
            "rouge_l_winner": "react" if react_agg["avg_rouge_l"] >= pe_agg["avg_rouge_l"] else "plan_execute",
            "latency_winner": "react" if react_agg["avg_latency_ms"] <= pe_agg["avg_latency_ms"] else "plan_execute",
            "tool_accuracy_winner": "react" if react_agg["avg_tool_accuracy"] >= pe_agg["avg_tool_accuracy"] else "plan_execute",
            "react_latency_advantage_pct": pct_diff(pe_agg["avg_latency_ms"], react_agg["avg_latency_ms"]),
            "react_accuracy_advantage_pct": pct_diff(react_agg["avg_tool_accuracy"], pe_agg["avg_tool_accuracy"]),
        }
    }
