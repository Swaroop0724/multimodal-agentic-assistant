"""
main.py
Entry point for the Multimodal Agentic Assistant.
Supports three modes: chat, benchmark, and ui.
"""

import argparse
import sys
from config.logging_config import setup_logging

logger = setup_logging()


def run_chat(model: str, agent_type: str):
    """Interactive chat mode with chosen agent."""
    from agents.react_agent import ReActAgent
    from agents.plan_execute_agent import PlanExecuteAgent

    logger.info(f"Starting chat | agent={agent_type} | model={model}")
    print(f"\n🤖 Multimodal Agentic Assistant")
    print(f"   Agent: {agent_type.upper()} | Model: {model}")
    print(f"   Type 'exit' to quit\n")
    print("-" * 60)

    AgentClass = ReActAgent if agent_type == "react" else PlanExecuteAgent
    agent = AgentClass(model_name=model)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye! 👋")
                break
            if not user_input:
                continue

            print(f"\n🔄 Thinking...")
            result = agent.run(user_input)
            print(f"\n🤖 Assistant: {result['output']}")

            if result.get("tool_calls"):
                print(f"\n   📦 Tools used: {', '.join(result['tool_calls'])}")
            if result.get("latency_ms"):
                print(f"   ⏱️  Latency: {result['latency_ms']:.0f}ms")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Chat error: {e}")
            print(f"\n❌ Error: {e}")


def run_benchmark(query_count: int):
    """Run full benchmark comparing ReAct vs Plan-Execute."""
    from benchmark.harness import BenchmarkHarness

    logger.info(f"Starting benchmark | queries={query_count}")
    harness = BenchmarkHarness(query_count=query_count)
    harness.run()


def run_ui():
    """Launch Streamlit UI."""
    import subprocess
    logger.info("Launching Streamlit UI")
    subprocess.run([
        "streamlit", "run", "ui/app.py",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal Agentic Assistant — ReAct vs Plan-Execute"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Run mode")

    # Chat mode
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--agent", choices=["react", "plan-execute"], default="react",
        help="Agent architecture (default: react)"
    )
    chat_parser.add_argument(
        "--model", default="llama-70b",
        choices=["llama-70b", "llama-8b", "mixtral"],
        help="LLM model (default: llama-70b)"
    )

    # Benchmark mode
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation")
    bench_parser.add_argument(
        "--queries", type=int, default=50,
        help="Number of queries to evaluate (default: 50)"
    )

    # UI mode
    subparsers.add_parser("ui", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.mode == "chat":
        run_chat(model=args.model, agent_type=args.agent)
    elif args.mode == "benchmark":
        run_benchmark(query_count=args.queries)
    elif args.mode == "ui":
        run_ui()
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python main.py chat              # Interactive chat with ReAct agent")
        print("   python main.py chat --agent plan-execute   # Plan-Execute agent")
        print("   python main.py benchmark         # Run 50-query benchmark")
        print("   python main.py ui                # Launch Streamlit dashboard")


if __name__ == "__main__":
    main()
