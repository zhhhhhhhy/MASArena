#!/usr/bin/env python3
"""
Benchmark Runner Script

This script provides a simplified way to run benchmarks with different configurations.
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

# Available benchmarks and agent systems
BENCHMARKS = ["math", "drop", "gsm8k", "hotpotqa", "humaneval", "mbpp"]
DEFAULT_AGENT_SYSTEMS = ["single_agent", "supervisor_mas", "swarm"]

def run_benchmark(benchmark="math", agent_system="single_agent", limit=10, 
                 results_dir="results", metrics_dir="metrics", verbose=True):
    """
    Run a benchmark with the specified configuration.
    
    Args:
        benchmark: The benchmark to run
        agent_system: The agent system to use
        limit: Maximum number of problems to process
        results_dir: Directory to store results
        metrics_dir: Directory to store metrics
        verbose: Whether to print progress information
        
    Returns:
        The return code from running the benchmark
    """
    # Create directories if needed
    Path(results_dir).mkdir(exist_ok=True)
    Path(metrics_dir).mkdir(exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "main.py",
        "--benchmark", benchmark,
        "--agent-system", agent_system,
        "--limit", str(limit),
        "--results-dir", results_dir,
        "--metrics-dir", metrics_dir
    ]
    
    if not verbose:
        cmd.remove("--verbose")
    
    # Run the benchmark
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)

def run_all_agent_systems(benchmark="math", limit=5, 
                         results_dir="results", metrics_dir="metrics"):
    """
    Run a benchmark with all available agent systems.
    
    Args:
        benchmark: The benchmark to run
        limit: Maximum number of problems to process
        results_dir: Directory to store results
        metrics_dir: Directory to store metrics
        
    Returns:
        Dictionary with results for each agent system
    """
    results = {}
    
    print(f"Running benchmark '{benchmark}' with all agent systems")
    print(f"Processing {limit} problems per agent system")
    print("=" * 60)
    
    for agent_system in DEFAULT_AGENT_SYSTEMS:
        print(f"\nRunning {agent_system}...")
        return_code = run_benchmark(
            benchmark=benchmark,
            agent_system=agent_system,
            limit=limit,
            results_dir=results_dir,
            metrics_dir=metrics_dir
        )
        results[agent_system] = return_code
        print(f"Completed {agent_system} with return code {return_code}")
        print("-" * 40)
    
    return results

def run_all_benchmarks(agent_system="single_agent", limit=5,
                      results_dir="results", metrics_dir="metrics"):
    """
    Run all benchmarks with a specific agent system.
    
    Args:
        agent_system: The agent system to use
        limit: Maximum number of problems to process
        results_dir: Directory to store results
        metrics_dir: Directory to store metrics
        
    Returns:
        Dictionary with results for each benchmark
    """
    results = {}
    
    print(f"Running all benchmarks with '{agent_system}'")
    print(f"Processing {limit} problems per benchmark")
    print("=" * 60)
    
    for benchmark in BENCHMARKS:
        print(f"\nRunning {benchmark}...")
        return_code = run_benchmark(
            benchmark=benchmark,
            agent_system=agent_system,
            limit=limit,
            results_dir=results_dir,
            metrics_dir=metrics_dir
        )
        results[benchmark] = return_code
        print(f"Completed {benchmark} with return code {return_code}")
        print("-" * 40)
    
    return results

def main():
    """Parse command-line arguments and run benchmarks"""
    parser = argparse.ArgumentParser(description="Run benchmarks with different configurations")
    
    parser.add_argument("--benchmark", type=str, default="math", choices=BENCHMARKS,
                        help=f"Benchmark to run (default: math)")
    
    parser.add_argument("--agent-system", type=str, default="single_agent", 
                        choices=DEFAULT_AGENT_SYSTEMS,
                        help=f"Agent system to use (default: single_agent)")
    
    parser.add_argument("--limit", type=int, default=10,
                        help="Maximum number of problems to process (default: 10)")
    
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to store results (default: results)")
    
    parser.add_argument("--metrics-dir", type=str, default="metrics",
                        help="Directory to store metrics (default: metrics)")
    
    parser.add_argument("--all-agents", action="store_true",
                        help="Run the benchmark with all agent systems")
    
    parser.add_argument("--all-benchmarks", action="store_true",
                        help="Run all benchmarks with the specified agent system")
    
    args = parser.parse_args()
    
    if args.all_agents:
        results = run_all_agent_systems(
            benchmark=args.benchmark,
            limit=args.limit,
            results_dir=args.results_dir,
            metrics_dir=args.metrics_dir
        )
        sys.exit(0 if all(code == 0 for code in results.values()) else 1)
    
    elif args.all_benchmarks:
        results = run_all_benchmarks(
            agent_system=args.agent_system,
            limit=args.limit,
            results_dir=args.results_dir,
            metrics_dir=args.metrics_dir
        )
        sys.exit(0 if all(code == 0 for code in results.values()) else 1)
    
    else:
        return_code = run_benchmark(
            benchmark=args.benchmark,
            agent_system=args.agent_system,
            limit=args.limit,
            results_dir=args.results_dir,
            metrics_dir=args.metrics_dir
        )
        sys.exit(return_code)

if __name__ == "__main__":
    main() 