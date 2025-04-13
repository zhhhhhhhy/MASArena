#!/usr/bin/env python3
import argparse
import datetime
import sys
from pathlib import Path

from benchmark.benchmark_runner import BenchmarkRunner
from benchmark.src.agents import AVAILABLE_AGENT_SYSTEMS


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run benchmarks for multi-agent systems")

    # Import available agent systems
    from benchmark.src.agents import AVAILABLE_AGENT_SYSTEMS

    parser.add_argument(
        "--benchmark",
        type=str,
        default="math",
        choices=["math", "drop", "gsm8k", "hotpotqa", "humaneval", "mbpp"],
        help="Benchmark to run (default: math)",
    )

    parser.add_argument(
        "--data", type=str, default=None, help="Path to benchmark data (default: benchmark/data/{benchmark}_test.jsonl)"
    )

    parser.add_argument("--limit", type=int, default=10, help="Maximum number of problems to process (default: 10)")

    parser.add_argument(
        "--agent-system",
        type=str,
        default="single_agent",
        choices=list(AVAILABLE_AGENT_SYSTEMS.keys()),
        help="Agent system to use (default: single_agent)",
    )

    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Print progress information (default: True)"
    )

    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directory to store results (default: results)"
    )

    parser.add_argument(
        "--metrics-dir", type=str, default="metrics", help="Directory to store metrics (default: metrics)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Create directories if needed
    Path(args.results_dir).mkdir(exist_ok=True)
    Path(args.metrics_dir).mkdir(exist_ok=True)

    # Print header
    print("\n" + "=" * 80)
    print(f"Multi-Agent Benchmark Runner ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 80)
    print(f"Benchmark: {args.benchmark}")
    print(f"Agent System: {args.agent_system}")
    print(f"Data: {args.data or 'default'}")
    print(f"Limit: {args.limit}")
    print(f"Results will be stored in: {args.results_dir}")
    print(f"Metrics will be stored in: {args.metrics_dir}")
    print("=" * 80 + "\n")

    # Create benchmark runner
    runner = BenchmarkRunner(results_dir=args.results_dir, metrics_dir=args.metrics_dir)

    # Run benchmark
    try:
        summary = runner.run(
            benchmark_name=args.benchmark,
            data_path=args.data,
            limit=args.limit,
            agent_system=args.agent_system,
            verbose=args.verbose,
        )

        print("\n" + "=" * 80)
        print("Benchmark Summary")
        print("=" * 80)
        print(f"Benchmark: {summary['benchmark']}")
        print(f"Agent System: {summary['agent_system']}")
        print(f"Problems: {summary['total_problems']}")
        print(f"Correct: {summary['correct']}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        print(f"Total Duration: {summary['total_duration_ms']:.2f} ms")
        print(f"Average Duration: {summary['avg_duration_ms']:.2f} ms")
        print(f"Results saved to: {summary['results_file']}")
        print(f"Metrics saved to: {summary['metrics_dir']}")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
