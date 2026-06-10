"""
benchmark/harness.py
Main benchmark orchestrator.

Runs N queries through BOTH ReAct and Plan-Execute agents,
computes all metrics, saves results to CSV + JSON.

Usage:
    python main.py benchmark --queries 50
    python main.py benchmark --queries 10   # quick smoke test
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from loguru import logger

from benchmark.metrics import (
    QueryMetrics,
    compute_rouge_l,
    compute_tool_accuracy,
    aggregate_metrics,
    compare_agents,
)
from config.settings import settings


class BenchmarkHarness:
    """
    Orchestrates the full benchmark run.

    For each query:
      1. Run ReAct agent  → collect AgentResult
      2. Run Plan-Execute → collect AgentResult
      3. Compute ROUGE-L + tool accuracy for both
      4. Log to CSV in real-time (so partial results are always saved)

    Final output:
      data/results/benchmark_<timestamp>.csv
      data/results/benchmark_<timestamp>_summary.json
    """

    def __init__(
        self,
        query_count: int = 50,
        models: list[str] = None,
        queries_path: str = None,
    ):
        self.query_count = query_count
        self.models = models or ["llama-70b"]   # Default: test with one model
        self.queries_path = queries_path or settings.benchmark_queries_dir + "/benchmark_queries.json"
        self.results_dir = Path(settings.benchmark_results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Timestamp for this run
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Storage
        self.react_metrics: list[QueryMetrics] = []
        self.pe_metrics: list[QueryMetrics] = []

    def _load_queries(self) -> list[dict]:
        """Load benchmark queries from JSON file."""
        with open(self.queries_path, "r") as f:
            all_queries = json.load(f)

        # Shuffle for variety, then take N
        import random
        random.seed(42)  # Reproducible
        random.shuffle(all_queries)
        queries = all_queries[:self.query_count]

        logger.info(f"Loaded {len(queries)} queries from {self.queries_path}")
        return queries

    def _run_single_query(
        self,
        agent,
        agent_type: str,
        model_name: str,
        query_data: dict,
    ) -> QueryMetrics:
        """
        Run a single query through one agent and return QueryMetrics.
        Catches all exceptions so one failure doesn't abort the benchmark.
        """
        query_id = query_data["id"]
        query = query_data["query"]
        reference = query_data["reference_answer"]
        expected_tool = query_data["expected_tool"]
        category = query_data["category"]

        try:
            result = agent.run(query)

            rouge_l = compute_rouge_l(result.output, reference)
            tool_acc = compute_tool_accuracy(result.tool_calls, expected_tool)

            return QueryMetrics(
                query_id=query_id,
                category=category,
                agent_type=agent_type,
                model_name=model_name,
                query=query,
                expected_tool=expected_tool,
                reference_answer=reference,
                actual_output=result.output[:500],  # Truncate for CSV
                tool_calls_made=result.tool_calls,
                latency_ms=result.latency_ms,
                success=result.success,
                rouge_l=rouge_l,
                tool_accuracy=tool_acc,
                error=result.error,
            )

        except Exception as e:
            logger.error(f"Query {query_id} failed: {e}")
            return QueryMetrics(
                query_id=query_id,
                category=category,
                agent_type=agent_type,
                model_name=model_name,
                query=query,
                expected_tool=expected_tool,
                reference_answer=reference,
                actual_output="",
                tool_calls_made=[],
                latency_ms=0.0,
                success=False,
                rouge_l=0.0,
                tool_accuracy=0.0,
                error=str(e),
            )

    def _init_csv(self, filepath: Path):
        """Write CSV header row."""
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "category", "agent_type", "model_name",
                "query", "expected_tool", "tool_calls_made",
                "rouge_l", "tool_accuracy", "latency_ms",
                "success", "error", "actual_output"
            ])

    def _append_csv(self, filepath: Path, m: QueryMetrics):
        """Append one result row to CSV (real-time write)."""
        with open(filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                m.query_id, m.category, m.agent_type, m.model_name,
                m.query[:100], m.expected_tool,
                "|".join(m.tool_calls_made),
                m.rouge_l, m.tool_accuracy, round(m.latency_ms, 1),
                m.success, m.error or "",
                m.actual_output[:200],
            ])

    def run(self):
        """
        Main benchmark loop.
        Runs all queries through both agents, saves results.
        """
        from agents.react_agent import ReActAgent
        from agents.plan_execute_agent import PlanExecuteAgent

        queries = self._load_queries()
        csv_path = self.results_dir / f"benchmark_{self.run_id}.csv"
        self._init_csv(csv_path)

        print(f"\n{'='*65}")
        print(f"  🚀 Benchmark Starting")
        print(f"  Queries : {len(queries)}")
        print(f"  Models  : {self.models}")
        print(f"  Output  : {csv_path}")
        print(f"{'='*65}\n")

        for model_name in self.models:
            print(f"\n📊 Model: {model_name}")
            print(f"  Initializing agents...")

            react_agent = ReActAgent(model_name=model_name)
            pe_agent = PlanExecuteAgent(model_name=model_name)

            # ── ReAct pass ─────────────────────────────────────────────────────
            print(f"\n  🔵 ReAct Agent ({len(queries)} queries)")
            for q in tqdm(queries, desc="  ReAct", ncols=70):
                m = self._run_single_query(react_agent, "react", model_name, q)
                self.react_metrics.append(m)
                self._append_csv(csv_path, m)

                # Brief rate-limit pause between API calls
                time.sleep(0.5)

            # ── Plan-Execute pass ───────────────────────────────────────────────
            print(f"\n  🟠 Plan-Execute Agent ({len(queries)} queries)")
            for q in tqdm(queries, desc="  PlanExec", ncols=70):
                m = self._run_single_query(pe_agent, "plan_execute", model_name, q)
                self.pe_metrics.append(m)
                self._append_csv(csv_path, m)

                time.sleep(0.5)

        # ── Compute final summary ───────────────────────────────────────────────
        comparison = compare_agents(self.react_metrics, self.pe_metrics)

        summary_path = self.results_dir / f"benchmark_{self.run_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(comparison, f, indent=2)

        self._print_summary(comparison)

        print(f"\n✅ Results saved:")
        print(f"   CSV     : {csv_path}")
        print(f"   Summary : {summary_path}")

        return comparison

    def _print_summary(self, comparison: dict):
        """Print a formatted benchmark summary to console."""
        r = comparison["react"]
        p = comparison["plan_execute"]
        c = comparison["comparison"]

        print(f"\n{'='*65}")
        print(f"  📊 BENCHMARK RESULTS")
        print(f"{'='*65}")
        print(f"  {'Metric':<30} {'ReAct':>10} {'Plan-Exec':>12}")
        print(f"  {'-'*54}")
        print(f"  {'ROUGE-L Score':<30} {r['avg_rouge_l']:>10.4f} {p['avg_rouge_l']:>12.4f}")
        print(f"  {'Tool Accuracy':<30} {r['avg_tool_accuracy']*100:>9.1f}% {p['avg_tool_accuracy']*100:>11.1f}%")
        print(f"  {'Avg Latency (ms)':<30} {r['avg_latency_ms']:>10.0f} {p['avg_latency_ms']:>12.0f}")
        print(f"  {'P95 Latency (ms)':<30} {r['p95_latency_ms']:>10.0f} {p['p95_latency_ms']:>12.0f}")
        print(f"  {'Success Rate':<30} {r['success_rate']*100:>9.1f}% {p['success_rate']*100:>11.1f}%")
        print(f"  {'Avg Tool Calls/Query':<30} {r['avg_tool_calls_per_query']:>10.2f} {p['avg_tool_calls_per_query']:>12.2f}")
        print(f"{'='*65}")
        print(f"\n  🏆 Winners:")
        print(f"     ROUGE-L      → {c['rouge_l_winner'].upper()}")
        print(f"     Latency      → {c['latency_winner'].upper()}")
        print(f"     Tool Accuracy → {c['tool_accuracy_winner'].upper()}")

        if c['react_latency_advantage_pct'] > 0:
            print(f"\n  ⚡ ReAct is {c['react_latency_advantage_pct']:.1f}% faster than Plan-Execute")
        if c['react_accuracy_advantage_pct'] > 0:
            print(f"  🎯 ReAct is {c['react_accuracy_advantage_pct']:.1f}% more accurate on tool selection")

        print(f"\n{'='*65}")
