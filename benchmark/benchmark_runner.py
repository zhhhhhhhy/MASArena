#!/usr/bin/env python3
"""
Simple Benchmark Runner Interface

This module provides a simplified interface for running benchmarks on multi-agent systems.
"""

import os
import json
import random
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, List
import numpy as np

from benchmark.src.metrics import (
    MetricsRegistry,
    SystemMetricsCollector,
    AgentMetricsCollector,
    InterAgentMetricsCollector,
    MetricsCollector
)
from benchmark.src.metrics.unified_evaluator import UnifiedEvaluator
from benchmark.src.agents import create_agent_system, AVAILABLE_AGENT_SYSTEMS
from benchmark.src.evaluators.utils.normalization import normalize_problem_keys

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
        registry.register_collector("system", SystemMetricsCollector())
        registry.register_collector("agent", AgentMetricsCollector())
        registry.register_collector("inter_agent", InterAgentMetricsCollector())
        return registry

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

        # Store agent configuration for later use
        self.agent_config = agent_config or {}

        # Determine data path
        if data_path is None:
            if benchmark_name.lower() == "aime":
                data_path = f"benchmark/data/{benchmark_name}_aime2025-i_test.jsonl"
            else:
                data_path = f"benchmark/data/{benchmark_name}_test.jsonl"

        output_file = Path(self.results_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}.json"
        metrics_output = Path(self.metrics_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}"

        if verbose:
            print(f"Running {benchmark_name} benchmark with {agent_system} agent system...")
            print(f"Using data from: {data_path}")
            print(f"Processing up to {limit} problems")

        # Start metrics collection
        self.metrics_registry.start_all_collectors()
        
        # Record benchmark metadata using centralized collector
        self.metrics_collector.record_metric(
            "benchmark.metadata",
            1.0,
            {
                "benchmark": benchmark_name,
                "agent_system": agent_system,
                "limit": limit,
                "timestamp": self.timestamp
            }
        )
        
        # Start benchmark timer through centralized collector
        self.metrics_collector.start_timer(
            "benchmark.execution",
            {
                "benchmark": benchmark_name,
                "agent_system": agent_system,
                "limit": limit
            }
        )

        # Create agent system with appropriate configuration
        if agent_config is None:
            agent_config = {}
            
        # Add evaluator name to configuration
        agent_config["evaluator"] = benchmark_name
        
        # Set mock_mcp flag if using mcp_config_file with "mock" in the name
        if agent_config.get("use_mcp_tools") and agent_config.get("mcp_config_file"):
            config_file = agent_config.get("mcp_config_file", "")
            if "mock" in config_file.lower():
                agent_config["mock_mcp"] = True
                if verbose:
                    print(f"Using mock MCP tools (config: {config_file})")

        # Create agent system
        agent = create_agent_system(agent_system, agent_config)
        if agent is None:
            raise ValueError(
                f"Unknown agent system: {agent_system}. Available: {', '.join(AVAILABLE_AGENT_SYSTEMS.keys())}"
            )

        # Set metrics registry
        agent.set_metrics_registry(self.metrics_registry)

        # Load problems
        try:
            with open(data_path, "r") as f:
                problems = [json.loads(line) for line in f]
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Limit problems
        # random sample
        problems = random.sample(problems, limit)
        
        # Record problem count
        self.metrics_collector.record_metric(
            "benchmark.problem_count",
            len(problems),
            {
                "benchmark": benchmark_name,
                "agent_system": agent_system
            }
        )

        # Process problems
        all_results = []
        correct_count = 0
        error_count = 0
        total_duration = 0

        for i, problem in enumerate(problems):
            # Normalize problem dictionary
            normalized_problem = normalize_problem_keys(problem, benchmark_name, i)

            problem_id = normalized_problem["id"]

            if verbose:
                print(f"\nProblem {i + 1}/{len(problems)}: {problem_id}")

            # Start problem timer
            self.metrics_collector.start_timer(
                f"benchmark.problem.{problem_id}",
                {
                    "problem_id": problem_id,
                    "benchmark": benchmark_name,
                    "agent_system": agent_system,
                    "problem_index": i
                }
            )

            try:
                # Evaluate the problem using the agent system
                results = agent.evaluate(normalized_problem, metrics_registry=self.metrics_registry)

                # Stop problem timer
                problem_duration_ms = self.metrics_collector.stop_timer(f"benchmark.problem.{problem_id}")
                
                duration_ms = results.get("execution_time_ms", problem_duration_ms)
                score = results.get("score", 0)
                is_correct = score == 1
                
                # Update statistics
                total_duration += duration_ms
                if is_correct:
                    correct_count += 1
                    
                # Record problem result
                self.metrics_collector.record_metric(
                    "benchmark.problem.result",
                    score,
                    {
                        "problem_id": problem_id,
                        "benchmark": benchmark_name,
                        "agent_system": agent_system,
                        "correct": is_correct,
                        "duration_ms": duration_ms
                    }
                )
                
                # Create result entry
                result_entry = {
                    "problem_id": problem_id,
                    "problem": normalized_problem["problem"],
                    "expected": normalized_problem["solution"],
                    "prediction": results.get("extracted_answer", ""),
                    "score": score,
                    "duration_ms": duration_ms,
                    "agent_system": agent_system,
                    "llm_usage": results.get("llm_usage", {}),  # Include LLM usage metrics
                    # Add a simplified summary for quick reference
                    "summary": {
                        "correct": is_correct,
                        "duration_ms": duration_ms,
                        "total_tokens": results.get("llm_usage", {}).get("total_tokens", 0)
                    },
                }

                all_results.append(result_entry)

                if verbose:
                    print(f"Result: {'✓' if is_correct else '✗'} ({duration_ms:.0f}ms)")
                    print(f"Expected: {normalized_problem['solution']}")
                    print(f"Predicted: {results.get('extracted_answer', '')}")

            except Exception as e:
                # Stop problem timer
                self.metrics_collector.stop_timer(f"benchmark.problem.{problem_id}")
                
                # Record error in metrics
                error_count += 1
                self.metrics_collector.record_error(
                    "problem_processing",
                    str(e),
                    {
                        "problem_id": problem_id,
                        "benchmark": benchmark_name,
                        "agent_system": agent_system,
                        "error_type": type(e).__name__
                    }
                )

                if verbose:
                    # print(f"Error: {e}")
                    print(f"Error processing problem {problem_id} in benchmark {benchmark_name}: {e}")

                all_results.append(
                    {
                        "problem_id": problem_id,
                        "problem": normalized_problem["problem"],
                        "error": str(e),
                        "duration_ms": 0,
                        "agent_system": agent_system,
                        # Add a simplified summary for error cases
                        "summary": {
                            "correct": False,
                            "duration_ms": 0,
                            "error": True,
                            "error_type": type(e).__name__,
                        },
                    }
                )

        # Stop benchmark timer
        benchmark_duration_ms = self.metrics_collector.stop_timer("benchmark.execution")
        
        # Record final benchmark statistics
        self.metrics_collector.record_system_metrics(
            {
                "total_problems": len(problems),
                "correct_count": correct_count,
                "error_count": error_count,
                "accuracy": correct_count / len(problems) if problems else 0,
                "total_duration_ms": total_duration,
                "avg_duration_per_problem": total_duration / len(problems) if problems else 0,
            },
            {
                "benchmark": benchmark_name,
                "agent_system": agent_system
            }
        )

        # Save results to file
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        # Export metrics
        self.metrics_registry.export_all("json", str(metrics_output))

        # Calculate statistics
        correct = sum(1 for r in all_results if r.get("score", 0) == 1)
        total = len(all_results)
        accuracy = correct / total if total > 0 else 0

        # Generate inference metrics using UnifiedEvaluator
        evaluator = UnifiedEvaluator()
        inference_metrics = evaluator.evaluate_inference_metrics([str(output_file)])
        
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
            "inference_metrics": inference_metrics.get(agent_system, {})
        }

        # Save summary with inference metrics to a separate file
        summary_file = Path(self.results_dir) / f"{benchmark_name}_{agent_system}_{self.timestamp}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Save results for visualization
        self.results = all_results
        self.summary = summary

        if verbose:
            print("\nBenchmark complete!")
            
            print("\n" + "=" * 80)
            print("Benchmark Summary")
            print("=" * 80)
            
            print(f"Agent system: {agent_system}")
            print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
            print(f"Total duration: {benchmark_duration_ms:.0f}ms")
            print(f"Results saved to: {output_file}")
            # print(f"Metrics saved to: {metrics_output}")
            print(f"Summary saved to: {summary_file}")
            print(f"Run visualization: $ python benchmark/src/visualization/visualize_benchmark.py visualize --summary {summary_file}")

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
    from benchmark.src.agents import AVAILABLE_AGENT_SYSTEMS

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
