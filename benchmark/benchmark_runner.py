#!/usr/bin/env python3
"""
Simple Benchmark Runner Interface

This module provides a simplified interface for running benchmarks on multi-agent systems.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from benchmark.src.metrics import (
    MetricsRegistry,
    SystemMetricsCollector,
    AgentMetricsCollector,
    InterAgentMetricsCollector,
    MetricsCollector
)
from benchmark.src.agents import create_agent_system, AVAILABLE_AGENT_SYSTEMS
from benchmark.src.evaluators import MathEvaluator


class BenchmarkRunner:
    """
    Simple interface for running benchmarks on multi-agent systems.

    Examples:
        >>> benchmark = BenchmarkRunner()
        >>> results = benchmark.run("math", limit=5)
        >>> benchmark.visualize_results()
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
        registry.register_collector("system", SystemMetricsCollector())
        registry.register_collector("agent", AgentMetricsCollector())
        registry.register_collector("inter_agent", InterAgentMetricsCollector())
        return registry
    
    def _get_evaluator(self, benchmark_name, data_path=None):
        """
        Get the appropriate evaluator for the benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            data_path: Custom path to data file
            
        Returns:
            An initialized evaluator
        """
        if benchmark_name.lower() == "math":
            return MathEvaluator(
                name=benchmark_name,
                config={
                    "data_path": data_path or f"benchmark/data/{benchmark_name}_test.jsonl",
                    "log_path": f"benchmark/data/results/{benchmark_name.upper()}"
                }
            )
        else:
            # Default to math evaluator for now
            # In the future, we would add more evaluator types based on benchmark_name
            return MathEvaluator(
                name=benchmark_name,
                config={
                    "data_path": data_path or f"benchmark/data/{benchmark_name}_test.jsonl",
                    "log_path": f"benchmark/data/results/{benchmark_name.upper()}"
                }
            )

    def run(
        self,
        benchmark_name="math",
        data_path=None,
        limit=10,
        agent_system="single_agent",
        agent_config=None,
        verbose=True,
    ):
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
        if verbose:
            print(f"Available agent systems: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}")
        
        # Determine data path
        if data_path is None:
            data_path = f"benchmark/data/{benchmark_name}_test.jsonl"

        output_file = Path(self.results_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}.json"
        metrics_output = Path(self.metrics_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}"

        if verbose:
            print(f"Running {benchmark_name} benchmark with {agent_system} agent system...")
            print(f"Using data from: {data_path}")
            print(f"Processing up to {limit} problems")

        # Load problems
        with open(data_path, "r") as f:
            problems = [json.loads(line) for line in f]

        # Limit problems
        problems = problems[:limit]

        # Start metrics collection
        self.metrics_registry.start_all_collectors()
        
        # Start benchmark timer
        self.metrics_collector.start_timer(
            "benchmark.execution",
            {"benchmark": benchmark_name, "agent_system": agent_system}
        )

        # Create agent system with appropriate configuration
        if agent_config is None:
            agent_config = {}
            
        # Add evaluator name to configuration
        agent_config["evaluator"] = benchmark_name
        # Create agent system
        agent = create_agent_system(agent_system, agent_config)
        if agent is None:
            raise ValueError(
                f"Unknown agent system: {agent_system}. Available: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}"
            )

        # Set metrics registry
        agent.set_metrics_registry(self.metrics_registry)

        # Process problems
        all_results = []

        for i, problem in enumerate(problems):
            # Add problem ID if not present
            if "id" not in problem:
                problem["id"] = f"problem_{i + 1}"

            problem_id = problem["id"]

            if verbose:
                print(f"\nProblem {i + 1}/{len(problems)}: {problem_id}")

            # Start problem timer
            self.metrics_collector.start_timer(
                "problem.execution",
                {"problem_id": problem_id, "benchmark": benchmark_name, "agent_system": agent_system}
            )

            try:
                # Evaluate the problem using the agent system
                results = agent.evaluate(problem, metrics_registry=self.metrics_registry)

                # Stop problem timer
                problem_duration_ms = self.metrics_collector.stop_timer("problem.execution")

                # Save results
                result_entry = {
                    "problem_id": problem_id,
                    "problem": problem["problem"],
                    "expected": problem["solution"],
                    "prediction": results.get("extracted_answer", ""),
                    "score": results.get("math_score", 0),
                    "token_usage": results.get("token_usage", {}),
                    "duration_ms": results.get("execution_time_ms", problem_duration_ms),
                    "agent_system": agent_system,
                    # Add a simplified summary for quick reference
                    "summary": {
                        "correct": results.get("math_score", 0) == 1,
                        "duration_ms": results.get("execution_time_ms", problem_duration_ms),
                        "total_tokens": sum(
                            tokens.get('total_tokens', tokens) if isinstance(tokens, dict) else tokens 
                            for tokens in results.get("token_usage", {}).values()
                        ),
                    },
                }

                all_results.append(result_entry)

                if verbose:
                    print(f"Result: {'✓' if result_entry['score'] == 1 else '✗'} ({result_entry['duration_ms']:.0f}ms)")

            except Exception as e:
                # Stop problem timer on error
                problem_duration_ms = self.metrics_collector.stop_timer("problem.execution")

                if verbose:
                    print(f"Error: {e}")

                # Record error in metrics
                self.metrics_collector.record_metric(
                    "problem.error",
                    1.0,
                    {
                        "problem_id": problem_id, 
                        "error": str(e)[:100], 
                        "agent_system": agent_system
                    }
                )

                all_results.append(
                    {
                        "problem_id": problem_id,
                        "problem": problem["problem"],
                        "error": str(e),
                        "duration_ms": problem_duration_ms,
                        "agent_system": agent_system,
                        # Add a simplified summary for error cases
                        "summary": {
                            "correct": False,
                            "duration_ms": problem_duration_ms,
                            "error": True,
                            "error_type": type(e).__name__,
                        },
                    }
                )

        # Stop benchmark timer
        benchmark_duration_ms = self.metrics_collector.stop_timer("benchmark.execution")

        # Save results to file
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Export metrics
        self.metrics_registry.export_all("json", str(metrics_output))

        # Calculate statistics
        correct = sum(1 for r in all_results if r.get("score", 0) == 1)
        total = len(all_results)
        accuracy = correct / total if total > 0 else 0

        # Summary
        summary = {
            "benchmark": benchmark_name,
            "agent_system": agent_system,
            "total_problems": total,
            "correct": correct,
            "accuracy": accuracy,
            "total_duration_ms": benchmark_duration_ms,
            "avg_duration_ms": benchmark_duration_ms / total if total > 0 else 0,
            "results_file": str(output_file),
            "metrics_dir": str(metrics_output),
            "timestamp": self.timestamp,
        }

        # Save results for visualization
        self.results = all_results
        self.summary = summary

        if verbose:
            print("\nBenchmark complete!")
            print(f"Agent system: {agent_system}")
            print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
            print(f"Total duration: {benchmark_duration_ms:.0f}ms")
            print(f"Results saved to: {output_file}")
            print(f"Metrics saved to: {metrics_output}")

        # Stop metrics collection
        self.metrics_registry.stop_all_collectors()

        return summary

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
    summary = benchmark.run(benchmark_name=benchmark_name, limit=limit, agent_system=agent_system)

    if visualize:
        benchmark.visualize_results()

    return summary


if __name__ == "__main__":
    import argparse

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
        choices=["single_agent", "supervisor_mas", "swarm"],
        help="Agent system to use",
    )
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of problems to process")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")

    args = parser.parse_args()

    summary = run_simple_benchmark(
        benchmark_name=args.benchmark, limit=args.limit, agent_system=args.agent_system, visualize=not args.no_viz
    )

    print(f"\nAccuracy: {summary['accuracy']:.2%}")
