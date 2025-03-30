#!/usr/bin/env python3
"""
Multi-Agent Benchmark Runner

This script serves as the main entry point for running benchmarks against
multi-agent systems. It coordinates the workload generation, instrumentation,
metrics collection, and evaluation.
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

# Import benchmark components
from benchmark.src.agents.supervisor_mas import evaluate_mas, create_mas_graph
from benchmark.src.evaluators.math import MATHBenchmark
from benchmark.src.metrics import (
    MetricsRegistry,
    SystemMetricsCollector,
    AgentMetricsCollector,
    InterAgentMetricsCollector
)


def initialize_metrics():
    """Initialize the metrics collection infrastructure"""
    registry = MetricsRegistry()
    
    # Set up system metrics collector
    system_collector = SystemMetricsCollector()
    registry.register_collector("system", system_collector)
    
    # Set up agent metrics collector
    agent_collector = AgentMetricsCollector()
    registry.register_collector("agent", agent_collector)
    
    # Set up inter-agent metrics collector
    inter_agent_collector = InterAgentMetricsCollector()
    registry.register_collector("inter_agent", inter_agent_collector)
    
    return registry


def main():
    """Main entry point for running benchmarks"""
    parser = argparse.ArgumentParser(description="Run multi-agent system benchmarks")
    
    # Add benchmark selection arguments
    parser.add_argument("--benchmark", type=str, choices=["math", "drop", "gsm8k", "hotpotqa", "humaneval", "mbpp"], 
                        help="Benchmark to run", default="math")
    parser.add_argument("--data-path", type=str, 
                        help="Path to benchmark data file", default=None)
    parser.add_argument("--limit", type=int, 
                        help="Limit number of problems to process", default=10)
    parser.add_argument("--output-dir", type=str, 
                        help="Directory to save results", default="results")
    parser.add_argument("--metrics-output", type=str,
                        help="Directory to save metrics data", default="metrics")
    
    args = parser.parse_args()
    
    # Set up appropriate paths
    if args.data_path is None:
        args.data_path = f"benchmark/src/data/{args.benchmark}_test.jsonl"
    
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.metrics_output, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(args.output_dir) / f"{args.benchmark}_results_{timestamp}.json"
    metrics_dir = Path(args.metrics_output) / f"{args.benchmark}_{timestamp}"
    
    print(f"Running {args.benchmark} benchmark...")
    print(f"Using data from: {args.data_path}")
    print(f"Results will be saved to: {output_file}")
    print(f"Metrics will be saved to: {metrics_dir}")
    
    # Initialize metrics collection
    metrics_registry = initialize_metrics()
    system_collector = metrics_registry.get_collector("system")
    agent_collector = metrics_registry.get_collector("agent")
    inter_agent_collector = metrics_registry.get_collector("inter_agent")
    
    # Start metrics collection
    metrics_registry.start_all_collectors()
    
    # Load problems
    with open(args.data_path, "r") as f:
        problems = [json.loads(line) for line in f]
    
    # Limit the number of problems if specified
    if args.limit > 0:
        problems = problems[:args.limit]
        
    print(f"Processing {len(problems)} problems")
    
    # Record benchmark start
    benchmark_start_time = time.time()
    system_collector.collect_point("benchmark.start", 1.0, {"benchmark": args.benchmark})
    
    # Run the benchmark
    all_results = []
    
    for i, problem in enumerate(problems):
        print(f"\nProcessing Problem {i + 1}/{len(problems)}")
        problem_id = f"problem_{i+1}"
        problem_desc = problem["problem"][:100]
        print(f"Problem: {problem_desc}...")
        
        # Record problem start
        problem_start_time = time.time()
        system_collector.collect_point(
            "problem.start",
            1.0,
            tags={"problem_id": problem_id, "benchmark": args.benchmark}
        )
        
        try:
            # Record each agent interaction during evaluation
            results = evaluate_mas(problem)
            
            # Record problem metrics
            problem_end_time = time.time()
            problem_duration_ms = (problem_end_time - problem_start_time) * 1000
            
            system_collector.record_latency(
                "problem.execution", 
                problem_duration_ms, 
                tags={"problem_id": problem_id, "benchmark": args.benchmark}
            )
            
            # Extract token usage for agent metrics
            token_usage = results.get("token_usage", {})
            for agent_id, tokens in token_usage.items():
                agent_collector.record_llm_usage(
                    agent_id=agent_id,
                    model_name="gpt-4o-mini",  # This should be dynamically determined in a real scenario
                    prompt_tokens=tokens,  # In supervisor_mas, these are already the total tokens
                    completion_tokens=0,   # We don't split tokens in supervisor_mas
                    latency_ms=problem_duration_ms / len(token_usage) if token_usage else 0,
                    tags={"problem_id": problem_id, "benchmark": args.benchmark}
                )
            
            # Record whether the answer was correct
            score = results.get("math_score", 0)
            system_collector.collect_point(
                metric_name="problem.accuracy",
                value=score,
                tags={"problem_id": problem_id, "benchmark": args.benchmark}
            )
            
            all_results.append({
                "problem": problem["problem"],
                "expected": problem["solution"],
                "prediction": results.get("extracted_answer", ""),
                "score": score,
                "token_usage": token_usage,
                "duration_ms": problem_duration_ms
            })
            
            print(f"Score: {score}, Duration: {problem_duration_ms:.2f}ms")
            
            # Record problem completion
            system_collector.collect_point(
                "problem.completion",
                1.0,
                tags={"problem_id": problem_id, "benchmark": args.benchmark, "success": True}
            )
            
        except Exception as e:
            problem_end_time = time.time()
            problem_duration_ms = (problem_end_time - problem_start_time) * 1000
            
            print(f"Error processing problem: {e}")
            all_results.append({
                "problem": problem["problem"],
                "error": str(e),
                "duration_ms": problem_duration_ms
            })
            
            # Record error
            system_collector.collect_point(
                "problem.error",
                1.0,
                tags={"problem_id": problem_id, "benchmark": args.benchmark, "error": str(e)[:100]}
            )
            
            # Record problem completion with failure
            system_collector.collect_point(
                "problem.completion",
                0.0,
                tags={"problem_id": problem_id, "benchmark": args.benchmark, "success": False}
            )
    
    # Record benchmark completion
    benchmark_end_time = time.time()
    benchmark_duration_ms = (benchmark_end_time - benchmark_start_time) * 1000
    system_collector.collect_point(
        "benchmark.completion", 
        1.0,
        tags={"benchmark": args.benchmark, "duration_ms": str(benchmark_duration_ms)}
    )
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Calculate aggregate statistics
    correct = sum(1 for r in all_results if r.get("score", 0) == 1)
    total = len(all_results)
    accuracy = correct / total if total > 0 else 0
    
    # Export all metrics
    metrics_registry.export_all("json", str(metrics_dir))
    
    # Stop metrics collection
    metrics_registry.stop_all_collectors()
    
    print(f"\nBenchmark complete!")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"Total duration: {benchmark_duration_ms:.2f}ms")
    print(f"Average problem time: {benchmark_duration_ms / total:.2f}ms")
    print(f"Results saved to: {output_file}")
    print(f"Metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    main() 