#!/usr/bin/env python3
"""
Simple Benchmark Runner Interface

This module provides a simplified interface for running benchmarks on multi-agent systems.
"""

import os
import json
import random
from pathlib import Path
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm
from openai.types.completion_usage import CompletionUsage
import traceback
from concurrent.futures import ThreadPoolExecutor

from mas_arena.src.metrics import (
    MetricsRegistry,
    MetricsCollector
)
from mas_arena.src.agents import create_agent_system, AVAILABLE_AGENT_SYSTEMS
from mas_arena.src.evaluators import BENCHMARKS
from mas_arena.src.evaluators.utils.normalization import normalize_problem_keys

def custom_json_serializer(obj):
    """Custom JSON serializer for objects that are not serializable by default."""
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    if hasattr(obj, '__dict__'):
        # For AIMessage-like objects, convert to dict
        return {key: getattr(obj, key) for key in obj.__dict__ if not key.startswith('_')}
    if isinstance(obj, CompletionUsage):
        return {
            "prompt_tokens": obj.prompt_tokens,
            "completion_tokens": obj.completion_tokens,
            "total_tokens": obj.total_tokens,
        }
    try:
        return str(obj)  # Fallback for other non-serializable types
    except Exception:
        return f"Object of type {type(obj).__name__} is not JSON serializable"

class BenchmarkRunner:
    """
    Simple interface for running benchmarks on multi-agent systems.

    Examples:
        >>> benchmark = BenchmarkRunner()
        >>> results = mas_arena.run("math", limit=5)
        >>> mas_arena.visualize_results()
    """

    def __init__(self, results_dir="results", metrics_dir="metrics"):
        """
        Initialize the benchmark runner.

        Args:
            results_dir: Directory to save results
            metrics_dir: Directory to save metrics data
        """
        self.results_dir = results_dir
        self.metrics_dir = metrics_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        self.agent_config = None  # Store agent configuration
        self.metrics_registry = None
        self.metrics_collector = None

        # Create directories
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        # Set up metrics
        self.metrics_registry = self._setup_metrics()
        
        # Create centralized metrics collector
        self.metrics_collector = MetricsCollector(self.metrics_registry)

    def _setup_metrics(self):
        """Set up metrics collection"""
        registry = MetricsRegistry()
        return registry

    def _prepare_benchmark(self, benchmark_name, data_path, limit, agent_system, agent_config, verbose):
        """
        Run a benchmark with the specified configuration.

        Args:
            benchmark_name: Name of the benchmark (math, drop, gsm8k, hotpotqa, humaneval, mbpp)
            data_path: Custom path to benchmark data file (optional)
            limit: Maximum number of problems to process
            agent_system: Agent system to use (single_agent, supervisor_mas, swarm)
            agent_config: Configuration for the agent system
            verbose: Whether to print progress information

        Returns:
            Dictionary of benchmark results
        """
        # Validate benchmark name
        if benchmark_name not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Supported: {', '.join(BENCHMARKS.keys())}")
        
        benchmark_config = BENCHMARKS[benchmark_name]

        if verbose:
            print(f"Available agent systems: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")

        self.agent_config = agent_config or {}

        # Ensure the agent knows which evaluator to use by setting it in the config
        self.agent_config['evaluator'] = benchmark_name

        if not data_path:
            data_path = benchmark_config.get("data_path", f"data/{benchmark_name}_test.jsonl")

        output_file = Path(self.results_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}.json"
        metrics_output = Path(self.metrics_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}"

        agent = create_agent_system(agent_system, self.agent_config)
        if agent is None:
            raise ValueError(f"Unknown agent system: {agent_system}. Available: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")

        agent.set_metrics_registry(self.metrics_registry)

        try:
            with open(data_path, "r") as f:
                problems = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        if limit and limit < len(problems):
            problems = random.sample(problems, limit)

        # Limit problems
        # random sample
        problems = random.sample(problems, limit)

        return agent, problems, benchmark_config, output_file, metrics_output

    def _process_one_problem(self, i, p, agent, benchmark_config, verbose=True):
        key_mapping = benchmark_config.get("normalization_keys", {})
        normalized_problem = normalize_problem_keys(p, key_mapping, i)
        problem_id = normalized_problem["id"]

        if verbose:
            print(f"\nProblem {i + 1}: {problem_id}")

        self.metrics_collector.start_timer(f"mas_arena.problem.{problem_id}", {"problem_id": problem_id})

        try:
            results = agent.evaluate(normalized_problem, metrics_registry=self.metrics_registry)
            problem_duration_ms = self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")

            duration_ms = results.get("execution_time_ms", problem_duration_ms)
            score = results.get("score", 0)
            is_correct = score == 1

            self.metrics_collector.record_metric("mas_arena.problem.result", score, {"problem_id": problem_id, "correct": is_correct, "duration_ms": duration_ms})

            result_entry = {
                "problem_id": problem_id,
                "problem": normalized_problem["problem"],
                "expected": normalized_problem["solution"],
                "prediction": results.get("extracted_answer", ""),
                "score": score,
                "duration_ms": duration_ms,
                "agent_system": agent.name,
                "llm_usage": results.get("llm_usage", {}),
                "summary": {"correct": is_correct, "score": score, "duration_ms": duration_ms},
            }
            if verbose:
                print(f"Result: {'✓' if is_correct else '✗'} ({duration_ms:.0f}ms)")
            return result_entry
        except Exception as e:
            self.metrics_collector.stop_timer(f"mas_arena.problem.{problem_id}")
            self.metrics_collector.record_error("problem_processing", str(e), {"problem_id": problem_id, "error_type": type(e).__name__})
            if verbose:
                print(f"Error processing problem {problem_id}: {e}")
                traceback.print_exc()
            return {"problem_id": problem_id, "problem": normalized_problem.get("problem"), "error": str(e)}

    def _finalize_benchmark(self, all_results, benchmark_name, agent_system, output_file, metrics_output, verbose):
        total = len(all_results)
        correct = sum(1 for r in all_results if r.get("score", 0) == 1)
        accuracy = correct / total if total > 0 else 0
        total_duration = sum(r.get("duration_ms", 0) for r in all_results)

        summary = {
            "benchmark": benchmark_name,
            "agent_system": agent_system,
            "total_problems": total,
            "correct": correct,
            "accuracy": accuracy,
            "total_duration_ms": total_duration,
            "avg_duration_ms": total_duration / total if total > 0 else 0,
            "results_file": str(output_file),
            "metrics_dir": str(metrics_output),
            "timestamp": self.timestamp,
        }

        self.metrics_registry.stop_all_collectors()
        self.metrics_registry.export_all(format="json", path=str(metrics_output))

        # Create a separate summary file for visualization
        summary_file = output_file.with_name(f"{output_file.stem}_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4, default=custom_json_serializer)

        # Save main results file
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "results": all_results}, f, indent=4, default=custom_json_serializer)

        if verbose:
            print("\n" + "=" * 80)
            print("Benchmark Summary")
            print("=" * 80)
            print(json.dumps(summary, indent=2))
            print("-" * 80)
            print("To visualize results, run:")
            print(f"$ python mas_arena/src/visualization/visualize_mas_arena.py visualize --summary {summary_file}")
            print("=" * 80)

        self.results = all_results
        self.summary = summary
        return summary

    def run(self, benchmark_name="math", data_path=None, limit=10, agent_system="single_agent", agent_config=None, verbose=True):
        agent, problems, benchmark_config, output_file, metrics_output = self._prepare_benchmark(
            benchmark_name, data_path, limit, agent_system, agent_config, verbose
        )

        if verbose:
            print(f"Running {benchmark_name} benchmark with {agent_system} agent system...")
            print(f"Processing {len(problems)} problems sequentially.")

        self.metrics_registry.start_all_collectors()
        self.metrics_collector.start_timer("mas_arena.execution")

        all_results = [
            self._process_one_problem(i, p, agent, benchmark_config, verbose)
            for i, p in enumerate(tqdm(problems, desc="Processing Problems"))
        ]

        return self._finalize_benchmark(all_results, benchmark_name, agent_system, output_file, metrics_output, verbose)

    async def arun(self, benchmark_name="math", data_path=None, limit=10, agent_system="single_agent", agent_config=None, verbose=True, concurrency=10):
        # The math benchmark uses a library that is not thread-safe.
        # As a pragmatic solution, we fall back to synchronous execution for it.
        if benchmark_name == "math":
            if verbose:
                print("Math benchmark does not support concurrency. Running synchronously.")
            return self.run(benchmark_name, data_path, limit, agent_system, agent_config, verbose)

        agent, problems, benchmark_config, output_file, metrics_output = self._prepare_benchmark(
            benchmark_name, data_path, limit, agent_system, agent_config, verbose
        )

        if verbose:
            print(f"Running {benchmark_name} benchmark asynchronously with {agent_system} agent system...")
            print(f"Processing {len(problems)} problems with concurrency {concurrency}.")

        self.metrics_registry.start_all_collectors()
        self.metrics_collector.start_timer("mas_arena.execution")

        all_results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, self._process_one_problem, i, p, agent, benchmark_config, verbose)
                for i, p in enumerate(problems)
            ]
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Problems"):
                result = await future
                all_results.append(result)

        return self._finalize_benchmark(all_results, benchmark_name, agent_system, output_file, metrics_output, verbose)

    def visualize_results(self, output_dir=None):
        """
        Generate visualizations from the benchmark results.

        Args:
            output_dir: Directory to save visualizations (defaults to metrics_dir/benchmark_timestamp/viz)
        """
        pass 


def run_simple_benchmark(benchmark_name="math", limit=5, agent_system="single_agent", visualize=True):
    """
    Simple one-line function to run a benchmark and get results.

    Args:
        benchmark_name: Name of the benchmark to run
        limit: Maximum number of problems to process
        agent_system: Agent system to use
        visualize: Whether to generate visualizations

    Returns:
        Benchmark summary dictionary
    """
    benchmark = BenchmarkRunner()
    summary = mas_arena.run(benchmark_name=benchmark_name, limit=limit, agent_system=agent_system, verbose=True)

    if visualize:
        mas_arena.visualize_results()

    return summary


if __name__ == "__main__":
    import argparse
    from mas_arena.src.agents import AVAILABLE_AGENT_SYSTEMS

    parser = argparse.ArgumentParser(description="Run a benchmark with a simplified interface")
    parser.add_argument(
        "--benchmark",
        default="math",
        choices=["math", "drop", "gsm8k", "hotpotqa", "humaneval", "mbpp"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--agent-system",
        default="single_agent",
        choices=list(AVAILABLE_AGENT_SYSTEMS.keys()),
        help="Agent system to use",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of problems to process")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")

    args = parser.parse_args()

    summary = run_simple_benchmark(
        benchmark_name=args.benchmark, 
        limit=args.limit, 
        agent_system=args.agent_system, 
        visualize=not args.no_viz
    )

    print(f"\nAccuracy: {summary['accuracy']:.2%}")
