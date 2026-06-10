"""
benchmark/reporter.py
Generates visual reports from benchmark CSV results.

Produces:
  - Bar charts: ROUGE-L, Tool Accuracy, Latency comparison
  - Per-category breakdown table
  - Resume-ready summary bullet

Usage:
    from benchmark.reporter import BenchmarkReporter
    reporter = BenchmarkReporter("data/results/benchmark_20241201_143022.csv")
    reporter.generate_all()
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from loguru import logger


def load_results_from_csv(csv_path: str) -> tuple[list, list]:
    """
    Load benchmark CSV and split into react / plan_execute result lists.
    Returns (react_rows, plan_execute_rows).
    """
    react_rows = []
    pe_rows = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["rouge_l"] = float(row["rouge_l"] or 0)
            row["tool_accuracy"] = float(row["tool_accuracy"] or 0)
            row["latency_ms"] = float(row["latency_ms"] or 0)
            row["success"] = row["success"].lower() == "true"

            if row["agent_type"] == "react":
                react_rows.append(row)
            else:
                pe_rows.append(row)

    return react_rows, pe_rows


def _avg(values: list) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def generate_text_report(csv_path: str) -> str:
    """
    Generate a plain-text report from benchmark results.
    Works without matplotlib — useful for CI/logging.
    """
    react_rows, pe_rows = load_results_from_csv(csv_path)

    def summarize(rows, name):
        successful = [r for r in rows if r["success"]]
        rouge_scores = [r["rouge_l"] for r in successful]
        tool_acc = [r["tool_accuracy"] for r in successful]
        latencies = [r["latency_ms"] for r in successful]

        # Per-category
        cats = {}
        for r in successful:
            cat = r["category"]
            if cat not in cats:
                cats[cat] = {"rouge": [], "acc": [], "lat": []}
            cats[cat]["rouge"].append(r["rouge_l"])
            cats[cat]["acc"].append(r["tool_accuracy"])
            cats[cat]["lat"].append(r["latency_ms"])

        return {
            "name": name,
            "total": len(rows),
            "success_rate": round(len(successful) / len(rows) * 100, 1) if rows else 0,
            "avg_rouge_l": _avg(rouge_scores),
            "avg_tool_accuracy": round(_avg(tool_acc) * 100, 1),
            "avg_latency_ms": round(_avg(latencies), 0),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 0) if latencies else 0,
            "categories": {
                cat: {
                    "rouge_l": _avg(v["rouge"]),
                    "tool_accuracy": round(_avg(v["acc"]) * 100, 1),
                    "latency_ms": round(_avg(v["lat"]), 0),
                    "count": len(v["rouge"]),
                }
                for cat, v in cats.items()
            }
        }

    r = summarize(react_rows, "ReAct")
    p = summarize(pe_rows, "Plan-Execute")

    # Latency advantage
    lat_adv = round((p["avg_latency_ms"] - r["avg_latency_ms"]) / p["avg_latency_ms"] * 100, 1) \
        if p["avg_latency_ms"] > 0 else 0
    acc_adv = round(r["avg_tool_accuracy"] - p["avg_tool_accuracy"], 1)

    lines = [
        "=" * 65,
        "  BENCHMARK REPORT — Multimodal Agentic Assistant",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        f"  {'Metric':<32} {'ReAct':>10} {'Plan-Exec':>12}",
        f"  {'-'*56}",
        f"  {'Total Queries':<32} {r['total']:>10} {p['total']:>12}",
        f"  {'Success Rate':<32} {r['success_rate']:>9.1f}% {p['success_rate']:>11.1f}%",
        f"  {'ROUGE-L Score':<32} {r['avg_rouge_l']:>10.4f} {p['avg_rouge_l']:>12.4f}",
        f"  {'Tool Accuracy':<32} {r['avg_tool_accuracy']:>9.1f}% {p['avg_tool_accuracy']:>11.1f}%",
        f"  {'Avg Latency (ms)':<32} {r['avg_latency_ms']:>10.0f} {p['avg_latency_ms']:>12.0f}",
        f"  {'P95 Latency (ms)':<32} {r['p95_latency_ms']:>10.0f} {p['p95_latency_ms']:>12.0f}",
        "",
        "=" * 65,
        "  PER-CATEGORY BREAKDOWN (ReAct)",
        "=" * 65,
        f"  {'Category':<20} {'ROUGE-L':>8} {'Tool Acc':>10} {'Latency':>10} {'Count':>7}",
        f"  {'-'*56}",
    ]
    for cat, vals in r["categories"].items():
        lines.append(
            f"  {cat:<20} {vals['rouge_l']:>8.4f} {vals['tool_accuracy']:>9.1f}% "
            f"{vals['latency_ms']:>9.0f}ms {vals['count']:>6}"
        )

    lines += [
        "",
        "=" * 65,
        "  RESUME BULLET (auto-generated)",
        "=" * 65,
        "",
        f"  \"Benchmarked ReAct vs Plan-and-Execute agent architectures",
        f"  across {r['total']} queries, where ReAct achieved tool-call",
        f"  accuracy {r['avg_tool_accuracy']}% vs {p['avg_tool_accuracy']}%",
        f"  with {lat_adv}% lower latency. Compared Groq Llama 3.1-70B",
        f"  vs 8B vs Mixtral across 5 tools (web search, code execution,",
        f"  vision, Wikipedia, calculator) — built with LangGraph, Tavily,",
        f"  and E2B sandbox.\"",
        "",
        "=" * 65,
    ]

    return "\n".join(lines)


class BenchmarkReporter:
    """
    High-level reporter — generates text report + saves to file.
    Plotly charts generated separately in Streamlit UI.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.output_dir = Path(csv_path).parent
        self.run_id = Path(csv_path).stem.replace("benchmark_", "")

    def generate_all(self):
        """Generate text report and save it."""
        # Text report
        report = generate_text_report(self.csv_path)
        print(report)

        report_path = self.output_dir / f"report_{self.run_id}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"Report saved to {report_path}")
        return report_path

    @staticmethod
    def get_latest_csv(results_dir: str = "data/results") -> str | None:
        """Find the most recent benchmark CSV file."""
        results_path = Path(results_dir)
        csv_files = sorted(results_path.glob("benchmark_*.csv"))
        # Filter out summary files
        csv_files = [f for f in csv_files if "summary" not in f.name]
        return str(csv_files[-1]) if csv_files else None
