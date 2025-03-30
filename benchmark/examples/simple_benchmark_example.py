#!/usr/bin/env python3
"""
Simple Benchmark Example

This script demonstrates how to use the simplified benchmark interface.
"""

from benchmark.benchmark_runner import Benchmark, run_simple_benchmark

def example_1_one_line():
    """Example 1: One-line benchmark runner"""
    print("Example 1: One-line benchmark runner")
    
    # Run a benchmark with just one line of code
    # This will automatically collect metrics and generate visualizations
    summary = run_simple_benchmark(
        benchmark_name="math",  # The benchmark dataset to use
        limit=2,                # Process only 2 problems to save time
        visualize=True          # Generate visualizations
    )
    
    # The summary contains key statistics
    print(f"Benchmark: {summary['benchmark']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Average duration: {summary['avg_duration_ms']:.0f}ms")
    
    print("\nResults and visualizations saved to:")
    print(f"- Results: {summary['results_file']}")
    print(f"- Metrics: {summary['metrics_dir']}")
    print(f"- Visualizations: metrics/viz_{summary['timestamp']}")
    
    print("\n" + "-" * 50 + "\n")
    return summary


def example_2_benchmark_class():
    """Example 2: Using the Benchmark class for more control"""
    print("Example 2: Using the Benchmark class")
    
    # Create a benchmark object
    benchmark = Benchmark(
        results_dir="custom_results",     # Custom directory for results
        metrics_dir="custom_metrics"      # Custom directory for metrics
    )
    
    # Run the benchmark
    summary = benchmark.run(
        benchmark_name="math",   # The benchmark dataset to use
        limit=2,                 # Process only 2 problems to save time
        verbose=True             # Print progress information
    )
    
    # Generate visualizations with a custom output directory
    viz_dir = benchmark.visualize_results(
        output_dir="custom_metrics/visualizations"
    )
    
    print("\nCustom directories:")
    print(f"- Results: {summary['results_file']}")
    print(f"- Metrics: {summary['metrics_dir']}")
    print(f"- Visualizations: {viz_dir}")
    
    return summary


def example_3_compare_benchmarks():
    """Example 3: Comparing different benchmarks"""
    print("Example 3: Comparing different benchmarks")
    
    # Create a benchmark object
    benchmark = Benchmark(
        results_dir="comparison_results",
        metrics_dir="comparison_metrics"
    )
    
    # Run two different benchmarks
    results = {}
    
    # Only run with 1 problem each to save time in this example
    for name in ["math", "gsm8k"]:
        try:
            print(f"\nRunning {name} benchmark...")
            results[name] = benchmark.run(
                benchmark_name=name,
                limit=1,
                verbose=True
            )
        except FileNotFoundError:
            print(f"Dataset for {name} not found. Skipping.")
    
    # Compare results
    if len(results) > 1:
        print("\nComparison:")
        for name, summary in results.items():
            print(f"{name}: {summary['accuracy']:.2%} accuracy, " 
                  f"{summary['avg_duration_ms']:.0f}ms avg duration")
    
    return results


if __name__ == "__main__":
    # Run the examples
    example_1_one_line()
    example_2_benchmark_class()
    example_3_compare_benchmarks() 