#!/usr/bin/env python3
"""
Example demonstrating the Unified Evaluation Framework.

This script shows how to use the UnifiedEvaluator to evaluate different agent systems
using benchmark results and inference metrics.
"""

import os
import json
import argparse


from benchmark.src.metrics.unified_evaluator import (
    UnifiedEvaluator, 
    UnifiedEvaluationConfig,
)

def create_sample_results(output_path):
    """
    Create sample benchmark result files for demonstration purposes.
    
    Args:
        output_path: Path to save the sample results
    """
    # Sample data for different agent systems
    swarm_data = [
        {
            "problem_id": "problem_1",
            "problem": "Find the smallest positive integer that is greater than $1$ and relatively prime to the product of the first 20 positive integers.",
            "expected": "Two numbers are relatively prime if they share no prime factors...",
            "prediction": "23",
            "score": 1,
            "duration_ms": 62142.31,
            "agent_system": "swarm",
            "llm_usage": {
                "total_tokens": 5708,
                "message_count": 4,
                "agent_usage": [
                    {
                        "agent_id": "agent_1",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 126,
                        "completion_tokens": 545,
                        "reasoning_tokens": 0,
                        "total_tokens": 671,
                        "latency_ms": 62142.31
                    },
                    {
                        "agent_id": "agent_2",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 123,
                        "completion_tokens": 1061,
                        "reasoning_tokens": 0,
                        "total_tokens": 1184,
                        "latency_ms": 31071.15
                    },
                    {
                        "agent_id": "agent_3",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 125,
                        "completion_tokens": 813,
                        "reasoning_tokens": 0,
                        "total_tokens": 938,
                        "latency_ms": 20714.10
                    },
                    {
                        "agent_id": "aggregator",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 2561,
                        "completion_tokens": 354,
                        "reasoning_tokens": 0,
                        "total_tokens": 2915,
                        "latency_ms": 15535.58
                    }
                ]
            },
            "summary": {
                "correct": True,
                "duration_ms": 62142.31,
                "total_tokens": 5708
            }
        },
        {
            "problem_id": "problem_2",
            "problem": "How many integer divisors does $7$ have?",
            "expected": "The factors of $7$ are $-7, -1, 1,$ and $7$, for a total of $\\boxed{4}$ factors.",
            "prediction": "2",
            "score": 0,
            "duration_ms": 37268.50,
            "agent_system": "swarm",
            "llm_usage": {
                "total_tokens": 2593,
                "message_count": 4,
                "agent_usage": [
                    {
                        "agent_id": "agent_1",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 94,
                        "completion_tokens": 240,
                        "reasoning_tokens": 0,
                        "total_tokens": 334,
                        "latency_ms": 37268.50
                    },
                    {
                        "agent_id": "agent_2",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 91,
                        "completion_tokens": 221,
                        "reasoning_tokens": 0,
                        "total_tokens": 312,
                        "latency_ms": 18634.25
                    }
                ]
            },
            "summary": {
                "correct": False,
                "duration_ms": 37268.50,
                "total_tokens": 2593
            }
        }
    ]
    
    agentverse_data = [
        {
            "problem_id": "problem_1",
            "problem": "Find the smallest positive integer that is greater than $1$ and relatively prime to the product of the first 20 positive integers.",
            "expected": "Two numbers are relatively prime if they share no prime factors...",
            "prediction": "23",
            "score": 1,
            "duration_ms": 34148.11,
            "agent_system": "agentverse",
            "llm_usage": {
                "total_tokens": 14052,
                "message_count": 10,
                "agent_usage": [
                    {
                        "agent_id": "recruiter_recruiter_001",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 293,
                        "completion_tokens": 256,
                        "reasoning_tokens": 0,
                        "total_tokens": 549,
                        "latency_ms": 34148.11
                    },
                    {
                        "agent_id": "expert_1",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 143,
                        "completion_tokens": 1000,
                        "reasoning_tokens": 0,
                        "total_tokens": 1143,
                        "latency_ms": 17074.05
                    }
                ]
            },
            "summary": {
                "correct": True,
                "duration_ms": 34148.11,
                "total_tokens": 14052
            }
        },
        {
            "problem_id": "problem_2",
            "problem": "How many integer divisors does $7$ have?",
            "expected": "The factors of $7$ are $-7, -1, 1,$ and $7$, for a total of $\\boxed{4}$ factors.",
            "prediction": "4",
            "score": 1,
            "duration_ms": 29832.14,
            "agent_system": "agentverse",
            "llm_usage": {
                "total_tokens": 9876,
                "message_count": 8,
                "agent_usage": [
                    {
                        "agent_id": "recruiter_recruiter_001",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 245,
                        "completion_tokens": 198,
                        "reasoning_tokens": 0,
                        "total_tokens": 443,
                        "latency_ms": 29832.14
                    },
                    {
                        "agent_id": "expert_1",
                        "model_name": "gpt-4o-mini",
                        "prompt_tokens": 132,
                        "completion_tokens": 876,
                        "reasoning_tokens": 0,
                        "total_tokens": 1008,
                        "latency_ms": 14916.07
                    }
                ]
            },
            "summary": {
                "correct": True,
                "duration_ms": 29832.14,
                "total_tokens": 9876
            }
        }
    ]
    
    # Save as JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    swarm_path = os.path.join(os.path.dirname(output_path), "math_swarm_20250413_113622.json")
    agentverse_path = os.path.join(os.path.dirname(output_path), "math_agentverse_20250413_141244.json")
    
    with open(swarm_path, 'w') as f:
        json.dump(swarm_data, f, indent=2)
    
    with open(agentverse_path, 'w') as f:
        json.dump(agentverse_data, f, indent=2)
    
    print(f"Sample benchmark results created at {os.path.dirname(output_path)}")
    return [swarm_path, agentverse_path]


def evaluate_inference_metrics(result_files, output_path, config=None):
    """
    Evaluate inference metrics from benchmark result files.
    
    Args:
        result_files: List of paths to benchmark result files
        output_path: Directory to save evaluation results
        config: Optional configuration for the evaluator
    """
    # Create evaluator
    evaluator = UnifiedEvaluator(config or UnifiedEvaluationConfig())
    
    # Evaluate inference metrics
    inference_metrics = evaluator.evaluate_inference_metrics(result_files)
    
    # Export results
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "inference_metrics.json"), "w") as f:
        json.dump(inference_metrics, f, indent=2)
    
    return inference_metrics


def compare_agent_systems(inference_metrics, output_path, config=None):
    """
    Compare different agent systems based on inference metrics.
    
    Args:
        inference_metrics: Dictionary mapping agent system names to metrics
        output_path: Directory to save comparison results
        config: Optional configuration for the evaluator
    """
    # Create evaluation config for utility calculations
    evaluation_config = config or UnifiedEvaluationConfig()
    
    # Calculate utility for each system based on inference metrics
    results = {}
    
    for system_name, metrics in inference_metrics.items():
        # Calculate utility using the metrics
        # U(S) = V_A * A(S) + V_T * P(S) - C_R * R(S) - C_T * T(S)
        accuracy = metrics.get('accuracy', 0.0)
        throughput = metrics.get('throughput', 0.0)
        latency_s = metrics.get('latency', 0.0) / 1000  # ms to seconds
        memory_gb = metrics.get('memory', 0.0) / 1024  # MB to GB
        
        utility = (
            evaluation_config.V_A * accuracy + 
            evaluation_config.V_T * throughput - 
            evaluation_config.C_R * memory_gb - 
            evaluation_config.C_T * latency_s
        )
        
        # Calculate efficiency ratio
        value = evaluation_config.V_A * accuracy + evaluation_config.V_T * throughput
        cost = evaluation_config.C_R * memory_gb + evaluation_config.C_T * latency_s
        efficiency_ratio = value / cost if cost > 0 else float('inf')
        
        # Store results
        results[system_name] = {
            'raw_data': metrics,
            'utility': utility,
            'efficiency_ratio': efficiency_ratio,
            'system_type': system_name
        }
    
    # Export results
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "system_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def experiment_with_hyperparameters(inference_metrics, output_path):
    """
    Experiment with different hyperparameter settings to see how they affect evaluation.
    
    Args:
        inference_metrics: Dictionary mapping agent system names to metrics
        output_path: Directory to save evaluation results
    """
    # Define different hyperparameter configurations to test
    hyperparameter_configs = [
        # Default configuration (balanced)
        UnifiedEvaluationConfig(
            V_A=1.0, C_R=0.001, C_T=0.1, V_T=0.5,
            results_dir=os.path.join(output_path, "balanced")
        ),
        # Accuracy-focused configuration
        UnifiedEvaluationConfig(
            V_A=5.0, C_R=0.001, C_T=0.1, V_T=0.5,
            results_dir=os.path.join(output_path, "accuracy_focused")
        ),
        # Throughput-focused configuration
        UnifiedEvaluationConfig(
            V_A=1.0, C_R=0.001, C_T=0.1, V_T=2.5,
            results_dir=os.path.join(output_path, "throughput_focused")
        ),
        # Resource-sensitive configuration
        UnifiedEvaluationConfig(
            V_A=1.0, C_R=0.01, C_T=0.1, V_T=0.5,
            results_dir=os.path.join(output_path, "resource_sensitive")
        ),
        # Latency-sensitive configuration
        UnifiedEvaluationConfig(
            V_A=1.0, C_R=0.001, C_T=0.5, V_T=0.5,
            results_dir=os.path.join(output_path, "latency_sensitive")
        )
    ]
    
    # Evaluate with each configuration
    results_summary = {}
    for config in hyperparameter_configs:
        # Create results directory
        config_name = os.path.basename(config.results_dir)
        os.makedirs(config.results_dir, exist_ok=True)
        
        # Run comparison with this config
        results = compare_agent_systems(inference_metrics, config.results_dir, config)
        
        # Store summary
        results_summary[config_name] = {
            "hyperparameters": {
                "V_A": config.V_A,
                "C_R": config.C_R,
                "C_T": config.C_T,
                "V_T": config.V_T
            },
            "best_system": max(results.items(), key=lambda x: x[1]['utility'])[0],
            "utility_values": {name: data['utility'] for name, data in results.items()},
            "efficiency_ratios": {name: data['efficiency_ratio'] for name, data in results.items()}
        }
    
    # Save summary
    with open(os.path.join(output_path, "hyperparameter_experiment_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary table
    print("\nHyperparameter Experiment Summary:")
    print("-" * 80)
    print(f"{'Configuration':<20} | {'V_A':>5} | {'C_R':>5} | {'C_T':>5} | {'V_T':>5} | {'Best System':<20}")
    print("-" * 80)
    
    for config_name, summary in results_summary.items():
        hp = summary['hyperparameters']
        print(f"{config_name:<20} | {hp['V_A']:>5.2f} | {hp['C_R']:>5.4f} | {hp['C_T']:>5.2f} | {hp['V_T']:>5.2f} | {summary['best_system']:<20}")
    
    return results_summary


def main():
    """Main function to demonstrate the unified evaluation framework."""
    parser = argparse.ArgumentParser(description="Unified Evaluation Framework Example")
    parser.add_argument("--result-files", nargs="+", help="Paths to benchmark result files")
    parser.add_argument("--output", type=str, default="results/evaluation", help="Directory to save results")
    parser.add_argument("--create-sample", action="store_true", help="Create sample benchmark results")
    parser.add_argument("--experiment", action="store_true", help="Run hyperparameter experiment")
    
    args = parser.parse_args()
    
    # Create sample benchmark results if requested or if no result files provided
    if args.create_sample or not args.result_files:
        result_files = create_sample_results(os.path.join(args.output, "sample"))
    else:
        result_files = args.result_files
    
    # Evaluate inference metrics
    print("\nEvaluating inference metrics...")
    inference_metrics = evaluate_inference_metrics(result_files, args.output)
    
    # Print inference metrics
    print("\nInference Metrics:")
    print("-" * 100)
    print(f"{'System':<15} | {'Accuracy':>8} | {'Throughput':>12} | {'Latency (ms)':>12} | {'Memory (MB)':>12}")
    print("-" * 100)
    
    for system_name, metrics in inference_metrics.items():
        accuracy = metrics.get('accuracy', 0.0)
        throughput = metrics.get('throughput', 0.0)
        latency = metrics.get('latency', 0.0)
        memory = metrics.get('memory', 0.0)
        
        print(f"{system_name:<15} | {accuracy:>8.2f} | {throughput:>12.4f} | {latency:>12.2f} | {memory:>12.2f}")
    
    # Print model details
    for system_name, metrics in inference_metrics.items():
        models = metrics.get('model', [])
        if models:
            print(f"\nModel details for {system_name}:")
            print(f"{'Model':<15} | {'Latency (ms)':>12} | {'Input Tokens':>12} | {'Output Tokens':>12}")
            print("-" * 70)
            
            for model in models:
                model_name = model.get('model_name', '')
                latency = model.get('latency', 0.0)
                input_tokens = model.get('input_token_count', 0)
                output_tokens = model.get('output_token_count', 0)
                
                print(f"{model_name:<15} | {latency:>12.2f} | {input_tokens:>12} | {output_tokens:>12}")
    
    # Compare agent systems
    print("\nComparing agent systems...")
    comparison_results = compare_agent_systems(inference_metrics, args.output)
    
    # Print comparison results
    print("\nSystem Comparison:")
    print("-" * 100)
    print(f"{'System':<15} | {'Utility':>10} | {'Efficiency':>10} | {'Accuracy':>8} | {'Throughput':>10} | {'Latency (ms)':>12} | {'Memory (MB)':>12}")
    print("-" * 100)
    
    for system_name, data in comparison_results.items():
        raw_data = data['raw_data']
        accuracy = raw_data.get('accuracy', 0.0)
        throughput = raw_data.get('throughput', 0.0)
        latency = raw_data.get('latency', 0.0)
        memory = raw_data.get('memory', 0.0)
        
        print(f"{system_name:<15} | {data['utility']:>10.2f} | {data['efficiency_ratio']:>10.2f} | {accuracy:>8.2f} | {throughput:>10.4f} | {latency:>12.2f} | {memory:>12.2f}")
    
    # Run hyperparameter experiment if requested
    if args.experiment:
        print("\nRunning hyperparameter experiment...")
        experiment_results = experiment_with_hyperparameters(inference_metrics, args.output)
    
    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main() 