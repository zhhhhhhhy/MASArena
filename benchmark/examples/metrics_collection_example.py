#!/usr/bin/env python3
"""
Metrics Collection Example

This script demonstrates how to use the benchmark framework with metrics collection.
It runs a simple benchmark and shows how to process and visualize the collected metrics.
"""

import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from benchmark.src.agents.supervisor_mas import evaluate_mas
from benchmark.src.metrics import (
    MetricsRegistry,
    SystemMetricsCollector,
    AgentMetricsCollector,
    InterAgentMetricsCollector
)


def initialize_metrics():
    """Initialize metrics infrastructure"""
    registry = MetricsRegistry()
    
    # System metrics
    system_collector = SystemMetricsCollector()
    registry.register_collector("system", system_collector)
    
    # Agent metrics
    agent_collector = AgentMetricsCollector()
    registry.register_collector("agent", agent_collector)
    
    # Inter-agent metrics
    inter_agent_collector = InterAgentMetricsCollector()
    registry.register_collector("inter_agent", inter_agent_collector)
    
    return registry


def run_simple_benchmark(metrics_registry, problems, limit=3):
    """Run a simple benchmark with a few problems"""
    problems = problems[:limit]
    
    # Start metrics collection
    system_collector = metrics_registry.get_collector("system")
    system_collector.record_event("benchmark.start", tags={"benchmark": "math_example"})
    
    all_results = []
    start_time = time.time()
    
    for i, problem in enumerate(problems):
        print(f"\nProcessing Problem {i+1}/{len(problems)}")
        
        # Record problem start
        problem_id = f"problem_{i+1}"
        system_collector.record_event(
            "problem.start", 
            tags={"problem_id": problem_id, "benchmark": "math_example"}
        )
        
        try:
            # Evaluate problem with metrics
            results = evaluate_mas(problem, metrics_registry=metrics_registry)
            
            all_results.append({
                "problem_id": problem_id,
                "score": results.get("math_score", 0),
                "token_usage": results.get("token_usage", {}),
                "execution_time_ms": results.get("execution_time_ms", 0)
            })
            
            # Record problem completion
            system_collector.record_event(
                "problem.completion",
                tags={"problem_id": problem_id, "benchmark": "math_example", "success": True}
            )
            
        except Exception as e:
            print(f"Error: {e}")
            
            # Record problem error
            system_collector.record_event(
                "problem.error",
                tags={"problem_id": problem_id, "benchmark": "math_example", "error": str(e)[:100]}
            )
    
    # Record benchmark completion
    end_time = time.time()
    benchmark_duration_ms = (end_time - start_time) * 1000
    system_collector.record_event(
        "benchmark.completion", 
        tags={"benchmark": "math_example", "duration_ms": benchmark_duration_ms}
    )
    
    return all_results, benchmark_duration_ms


def analyze_metrics(metrics_registry, results, output_dir):
    """Analyze and visualize the collected metrics"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export all metrics to JSON
    metrics_registry.export_all("json", output_dir)
    
    # Get collectors
    system_collector = metrics_registry.get_collector("system")
    agent_collector = metrics_registry.get_collector("agent")
    
    # 1. Create execution time chart
    execution_times = [r.get("execution_time_ms", 0) for r in results]
    problem_ids = [r.get("problem_id", f"Problem {i+1}") for i, r in enumerate(results)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(problem_ids, execution_times)
    plt.title("Problem Execution Times")
    plt.xlabel("Problem")
    plt.ylabel("Execution Time (ms)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "execution_times.png"))
    plt.close()
    
    # 2. Create token usage chart
    agent_names = []
    token_counts = []
    
    for result in results:
        token_usage = result.get("token_usage", {})
        for agent, tokens in token_usage.items():
            agent_names.append(f"{agent} ({result['problem_id']})")
            token_counts.append(tokens)
    
    plt.figure(figsize=(12, 6))
    plt.bar(agent_names, token_counts)
    plt.title("Token Usage by Agent and Problem")
    plt.xlabel("Agent (Problem)")
    plt.ylabel("Token Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "token_usage.png"))
    plt.close()
    
    # 3. Create agent performance comparison
    agent_metrics = {}
    for result in results:
        token_usage = result.get("token_usage", {})
        for agent, tokens in token_usage.items():
            if agent not in agent_metrics:
                agent_metrics[agent] = {"tokens": 0, "problems": 0}
            agent_metrics[agent]["tokens"] += tokens
            agent_metrics[agent]["problems"] += 1
    
    # Calculate average tokens per problem for each agent
    agent_names = list(agent_metrics.keys())
    avg_tokens = [agent_metrics[agent]["tokens"] / agent_metrics[agent]["problems"] 
                  for agent in agent_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(agent_names, avg_tokens)
    plt.title("Average Token Usage by Agent")
    plt.xlabel("Agent")
    plt.ylabel("Average Tokens per Problem")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_tokens_by_agent.png"))
    plt.close()
    
    # Create summary report
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write("# Metrics Summary\n\n")
        f.write(f"Total problems processed: {len(results)}\n")
        f.write(f"Total execution time: {sum(execution_times):.2f} ms\n")
        f.write(f"Average execution time: {np.mean(execution_times):.2f} ms\n\n")
        
        f.write("## Agent Token Usage\n\n")
        for agent, metrics in agent_metrics.items():
            f.write(f"Agent: {agent}\n")
            f.write(f"  Total tokens: {metrics['tokens']}\n")
            f.write(f"  Problems processed: {metrics['problems']}\n")
            f.write(f"  Average tokens per problem: {metrics['tokens'] / metrics['problems']:.2f}\n\n")
    
    print(f"Metrics analysis saved to {output_dir}")


def main():
    """Main function"""
    # Load a sample of math problems
    data_path = "benchmark/data/math_test.jsonl"
    with open(data_path, "r") as f:
        problems = [json.loads(line) for line in f]
    
    # Initialize metrics
    metrics_registry = initialize_metrics()
    metrics_registry.start_all_collectors()
    
    # Run benchmark
    print(f"Running benchmark with {min(3, len(problems))} problems...")
    results, duration_ms = run_simple_benchmark(metrics_registry, problems, limit=3)
    
    # Analyze collected metrics
    output_dir = "metrics_example_output"
    analyze_metrics(metrics_registry, results, output_dir)
    
    # Stop metrics collection
    metrics_registry.stop_all_collectors()
    
    print("\nBenchmark complete!")
    print(f"Total duration: {duration_ms:.2f} ms")
    print(f"Check {output_dir} for metrics analysis and visualizations")


if __name__ == "__main__":
    main() 