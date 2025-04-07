#!/usr/bin/env python3
"""
Example demonstrating the Unified Evaluation Framework.

This script shows how to use the UnifiedEvaluator to evaluate different agent systems
using leaderboard data as instrumentation.
"""

import os
import json
import argparse


from benchmark.src.metrics.unified_evaluator import (
    UnifiedEvaluator, 
    UnifiedEvaluationConfig,
)
def create_sample_leaderboard(output_path):
    """
    Create a sample leaderboard file for demonstration purposes.
    
    Args:
        output_path: Path to save the sample leaderboard
    """
    # Sample data for different agent systems with model details
    sample_data = {
        "single_agent_gpt4": {
            "system_type": "single_agent",
            "accuracy": 0.85,
            "model": [{
                "model_name": "gpt4",
                "latency": 120000,  # ms
                "throughput": 0.5,  # tasks per minute
                "input_token": 0.00005,  # $ per token
                "output_token": 0.00015,  # $ per token
                "parameters": 175000000000,  # 175B parameters
                "activated_parameters": 175000000000
            }]
        },
        "supervisor_mas": {
            "system_type": "mas",
            "accuracy": 0.89,
            "model": [{
                "model_name": "gpt4",
                "latency": 100000,  # ms
                "throughput": 0.4,  # tasks per minute
                "input_token": 0.00005,  # $ per token
                "output_token": 0.00015,  # $ per token
                "parameters": 175000000000,  # 175B parameters
                "activated_parameters": 175000000000
            }, {
                "model_name": "gpt35",
                "latency": 50000,  # ms
                "throughput": 0.6,  # tasks per minute
                "input_token": 0.000015,  # $ per token
                "output_token": 0.00002,  # $ per token
                "parameters": 175000000,  # 175M parameters
                "activated_parameters": 175000000
            }, {
                "model_name": "gpt35",
                "latency": 50000,  # ms
                "throughput": 0.6,  # tasks per minute
                "input_token": 0.000015,  # $ per token
                "output_token": 0.00002,  # $ per token
                "parameters": 175000000,  # 175M parameters
                "activated_parameters": 175000000
            }]
        },
        "swarm_3agents": {
            "system_type": "swarm",
            "accuracy": 0.92,
            "model": [{
                "model_name": "gpt4",
                "latency": 120000,  # ms
                "throughput": 0.5,  # tasks per minute
                "input_token": 0.00005,  # $ per token
                "output_token": 0.00015,  # $ per token
                "parameters": 175000000000,  # 175B parameters
                "activated_parameters": 175000000000
            }, {
                "model_name": "gpt4",
                "latency": 120000,  # ms
                "throughput": 0.5,  # tasks per minute
                "input_token": 0.00005,  # $ per token
                "output_token": 0.00015,  # $ per token
                "parameters": 175000000000,  # 175B parameters
                "activated_parameters": 175000000000
            }, {
                "model_name": "gpt4",
                "latency": 120000,  # ms
                "throughput": 0.5,  # tasks per minute
                "input_token": 0.00005,  # $ per token
                "output_token": 0.00015,  # $ per token
                "parameters": 175000000000,  # 175B parameters
                "activated_parameters": 175000000000
            }]
        },
        "single_agent_gpt35": {
            "system_type": "single_agent",
            "accuracy": 0.72,
            "model": [{
                "model_name": "gpt35",
                "latency": 75000,  # ms
                "throughput": 0.8,  # tasks per minute
                "input_token": 0.000015,  # $ per token
                "output_token": 0.00002,  # $ per token
                "parameters": 175000000,  # 175M parameters
                "activated_parameters": 175000000
            }]
        }
    }
    
    # Save as JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample leaderboard data created at {output_path}")
    return sample_data


def evaluate_from_leaderboard(leaderboard_path, output_path, config=None):
    """
    Evaluate agent systems using data from a leaderboard file.
    
    Args:
        leaderboard_path: Path to the leaderboard data file
        output_path: Directory to save evaluation results
        config: Optional configuration for the evaluator
    """
    # Create evaluator
    evaluator = UnifiedEvaluator(config or UnifiedEvaluationConfig())
    
    # Load and evaluate leaderboard data
    results = evaluator.evaluate_from_leaderboard(leaderboard_path)
    
    # Export results
    os.makedirs(output_path, exist_ok=True)
    evaluator.export_results(results, os.path.join(output_path, "unified_evaluation_results.json"))
    
    # Visualize results
    if config and config.visualize_results:
        evaluator.visualize_results(results, output_path)
    
    return results


def evaluate_with_metrics_registry(leaderboard_path, output_path, config=None):
    """
    Demonstrate collecting metrics from leaderboard into a MetricsRegistry,
    then evaluating using the registry data.
    
    Args:
        leaderboard_path: Path to the leaderboard data file
        output_path: Directory to save evaluation results
        config: Optional configuration for the evaluator
    """
    try:
        # Load leaderboard data first
        with open(leaderboard_path, 'r') as f:
            leaderboard_data = json.load(f)
        
        # Create a simple, direct approach that doesn't rely on the complex collectors
        # This is more reliable for demonstration purposes
        results = {}
        
        for system_name, system_data in leaderboard_data.items():
            # Extract system metrics directly
            if 'model' in system_data:
                # Create evaluator to extract system metrics
                evaluator = UnifiedEvaluator(config or UnifiedEvaluationConfig())
                system_metrics = evaluator._extract_system_metrics(system_data)
                system_metrics['model'] = system_data.get('model', [])
            else:
                system_metrics = system_data
            
            # Calculate metrics using the UnifiedEvaluator
            evaluator = UnifiedEvaluator(config or UnifiedEvaluationConfig())
            utility = evaluator.calculate_utility(system_data)
            efficiency = evaluator.calculate_efficiency_ratio(system_data)
            
            # Store results
            results[system_name] = {
                'raw_data': system_metrics,
                'utility': utility,
                'efficiency_ratio': efficiency,
                'system_type': system_metrics.get('system_type', 'unknown')
            }
        
        # Export results
        os.makedirs(output_path, exist_ok=True)
        
        # Create an evaluator just for exporting and visualizing
        evaluator = UnifiedEvaluator(config or UnifiedEvaluationConfig())
        evaluator.export_results(results, os.path.join(output_path, "registry_evaluation_results.json"))
        
        # Visualize results
        if config and config.visualize_results:
            evaluator.visualize_results(results, output_path)
        
        return results
        
    except Exception as e:
        print(f"\nError in evaluate_with_metrics_registry: {str(e)}")
        print("Falling back to direct leaderboard evaluation for comparison...")
        
        # Fall back to direct evaluation as a comparison
        return evaluate_from_leaderboard(leaderboard_path, output_path, config)


def experiment_with_hyperparameters(leaderboard_path, output_path):
    """
    Experiment with different hyperparameter settings to see how they affect evaluation.
    
    Args:
        leaderboard_path: Path to the leaderboard data file
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
        
        # Run evaluation
        evaluator = UnifiedEvaluator(config)
        results = evaluator.evaluate_from_leaderboard(leaderboard_path)
        
        # Save results
        evaluator.export_results(results, os.path.join(config.results_dir, "results.json"))
        evaluator.visualize_results(results, config.results_dir)
        
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
    parser.add_argument("--leaderboard", type=str, help="Path to leaderboard data file")
    parser.add_argument("--output", type=str, default="benchmark/examples/output", help="Directory to save results")
    parser.add_argument("--create-sample", action="store_true", help="Create sample leaderboard data")
    parser.add_argument("--experiment", action="store_true", help="Run hyperparameter experiment")
    
    args = parser.parse_args()
    
    # Create sample leaderboard if requested or if no leaderboard path provided
    if args.create_sample or not args.leaderboard:
        sample_path = os.path.join(args.output, "sample_leaderboard.json")
        create_sample_leaderboard(sample_path)
        leaderboard_path = sample_path
    else:
        leaderboard_path = args.leaderboard
    
    # Basic evaluation
    print("\nRunning basic evaluation...")
    config = UnifiedEvaluationConfig(
        V_A=1.0,  # Value per Accuracy Unit
        C_R=0.001,  # Cost per Resource Unit
        C_T=0.1,  # Cost per Time Unit
        V_T=0.5,  # Value per Throughput Unit
        visualize_results=True
    )
    results = evaluate_from_leaderboard(leaderboard_path, args.output, config)
    
    # Print basic results
    print("\nEvaluation Results:")
    print("-" * 100)
    print(f"{'System':<20} | {'Utility':>10} | {'Efficiency':>10} | {'Accuracy':>8} | {'Throughput':>10} | {'Latency (s)':>12} | {'Token Cost ($)':>14}")
    print("-" * 100)
    
    for system_name, data in results.items():
        accuracy = data['raw_data'].get('accuracy', 0.0)
        throughput = data['raw_data'].get('throughput', 0.0)
        latency = data['raw_data'].get('latency', 0.0) / 1000  # ms to seconds
        token_cost = data['raw_data'].get('token_cost', 0.0)
        
        print(f"{system_name:<20} | {data['utility']:>10.2f} | {data['efficiency_ratio']:>10.2f} | {accuracy:>8.2f} | {throughput:>10.2f} | {latency:>12.2f} | {token_cost:>14.4f}")
    
    # Print model details for multi-agent systems
    for system_name, data in results.items():
        if data['system_type'] in ['mas', 'swarm'] and 'models' in data['raw_data']:
            print(f"\nModel details for {system_name}:")
            print(f"{'Model':<10} | {'Latency (s)':>12} | {'Throughput':>10} | {'Input Cost ($)':>14} | {'Output Cost ($)':>14}")
            print("-" * 70)
            
            models = data['raw_data'].get('models', [])
            for i, model in enumerate(models):
                model_name = model.get('model_name', f"Model-{i}")
                latency = model.get('latency', 0.0) / 1000  # ms to seconds
                throughput = model.get('throughput', 0.0)
                input_cost = model.get('input_token', 0.0) 
                output_cost = model.get('output_token', 0.0) 
                
                print(f"{model_name:<10} | {latency:>12.2f} | {throughput:>10.2f} | {input_cost:>14.4f} | {output_cost:>14.4f}")
    
    # Demonstrate using metrics registry
    print("\nEvaluating using metrics registry...")
    registry_results = evaluate_with_metrics_registry(leaderboard_path, args.output, config)
    
    # Run hyperparameter experiment if requested
    if args.experiment:
        print("\nRunning hyperparameter experiment...")
        experiment_results = experiment_with_hyperparameters(leaderboard_path, args.output)
    
    print(f"\nAll results saved to {args.output}")


if __name__ == "__main__":
    main() 