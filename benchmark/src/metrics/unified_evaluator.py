"""
Unified Evaluation Framework for Single-Agent and Multi-Agent Systems.

This module implements the unified evaluation framework described in the benchmark 
documentation, using leaderboard data as instrumentation for evaluation.
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from benchmark.src.metrics.collectors import BaseMetricsCollector, MetricsCollectionConfig
from benchmark.src.metrics.system_metrics import SystemMetricsCollector


@dataclass
class UnifiedEvaluationConfig:
    """Configuration for the unified evaluation framework."""
    
    # Hyperparameters for the unified utility function
    V_A: float = 1.0  # Value per Accuracy Unit ($/accuracy unit)
    C_R: float = 0.001  # Cost per Resource Unit ($/compute unit)
    C_T: float = 0.1  # Cost per Time Unit ($/time unit)
    V_T: float = 0.5  # Value per Throughput Unit ($/task/time unit)
    
    # Resource metrics configuration
    compute_unit_normalization: float = 1e6  # Normalize compute units (e.g., parameter count in millions)
    time_unit_normalization: float = 1000.0  # Normalize time units (e.g., ms to seconds)
 
    # Output configuration
    results_dir: str = "results"
    visualize_results: bool = True


class UnifiedEvaluator:
    """
    Implements the unified evaluation framework for single-agent and multi-agent systems.
    
    This evaluator calculates utility functions, efficiency ratios, and other metrics
    based on the leaderboard data, using the hyperparameters defined in the configuration.
    """
    
    def __init__(self, config: Optional[UnifiedEvaluationConfig] = None):
        """
        Initialize the unified evaluator with the specified configuration.
        
        Args:
            config: Configuration for the evaluator
        """
        self.config = config or UnifiedEvaluationConfig()
        
    def _extract_system_metrics(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract system-level metrics from the model-based data format.
        
        Args:
            system_data: Dictionary containing system performance data with model details
            
        Returns:
            Dictionary with aggregated system metrics
        """
        # Get system-level attributes
        accuracy = system_data.get('accuracy', 0.0)
        system_type = system_data.get('system_type', 'unknown')
        models = system_data.get('model', [])
        
        if not models:
            return {
                'accuracy': accuracy,
                'throughput': 0.0,
                'latency': 0.0,
                'resource_usage': 0.0,
                'token_cost': 0.0,
                'system_type': system_type
            }
        
        # Calculate system-level metrics based on models
        # For latency: 
        #  - Single agent: latency of the model
        #  - MAS with supervisor: sum of supervisor latency + max worker latency
        #  - Swarm: max latency among models
        if system_type == 'single_agent':
            latency = models[0].get('latency', 0.0)
            throughput = models[0].get('throughput', 0.0)
        elif system_type == 'mas':
            # Assume first model is supervisor, rest are workers
            if len(models) > 1:
                supervisor_latency = models[0].get('latency', 0.0)
                worker_latencies = [m.get('latency', 0.0) for m in models[1:]]
                latency = supervisor_latency + max(worker_latencies) if worker_latencies else supervisor_latency
                # Throughput is limited by the slowest component
                throughputs = [m.get('throughput', 0.0) for m in models]
                throughput = min(throughputs) if throughputs else 0.0
            else:
                latency = models[0].get('latency', 0.0)
                throughput = models[0].get('throughput', 0.0)
        elif system_type == 'swarm':
            # For swarm, latency is the maximum among all models
            latency = max(m.get('latency', 0.0) for m in models)
            # Throughput increases with more agents but with diminishing returns
            # We'll use a simple logarithmic scaling for this example
            base_throughput = models[0].get('throughput', 0.0) if models else 0.0
            throughput = base_throughput * (1 + 0.5 * np.log(len(models))) if base_throughput > 0 else 0.0
        else:
            # Default case
            latency = sum(m.get('latency', 0.0) for m in models)
            throughputs = [m.get('throughput', 0.0) for m in models]
            throughput = np.mean(throughputs) if throughputs else 0.0
        
        # Calculate total parameters and activated parameters
        total_params = sum(m.get('parameters', 0.0) for m in models)
        total_activated_params = sum(m.get('activated_parameters', 0.0) for m in models)
        
        # Calculate token cost based on input/output token prices
        total_token_cost = 0.0
        for model in models:
            input_token_count = model.get('input_token_count', 0.0)
            output_token_count = model.get('output_token_count', 0.0)
            total_token_cost += input_token_count + output_token_count
        
        return {
            'accuracy': accuracy,
            'throughput': throughput,
            'latency': latency,
            'resource_usage': total_activated_params,  # Using activated parameters as resource usage
            'total_parameters': total_params,
            'token_cost': total_token_cost,
            'system_type': system_type
        }
    
    def calculate_utility(self, system_data: Dict[str, Any]) -> float:
        """
        Calculate utility function for a system using the unified framework.
        
        U(S) = V_A * A(S) + V_T * P(S) - C_R * R(S) - C_T * T(S) 
        
        Args:
            system_data: Dictionary containing system performance data
                
        Returns:
            The calculated utility value
        """
        # Extract or aggregate system-level metrics if needed
        if 'model' in system_data:
            metrics = self._extract_system_metrics(system_data)
        else:
            metrics = system_data
        
        # Extract metrics
        accuracy = metrics.get('accuracy', 0.0)
        throughput = metrics.get('throughput', 0.0)
        resource_usage = metrics.get('resource_usage', 0.0) / self.config.compute_unit_normalization
        latency = metrics.get('latency', 0.0) / self.config.time_unit_normalization
        
        # Calculate utility
        utility = (self.config.V_A * accuracy + 
                  self.config.V_T * throughput - 
                  self.config.C_R * resource_usage - 
                  self.config.C_T * latency)  # Direct cost component
        
        return utility
    
    def calculate_efficiency_ratio(self, system_data: Dict[str, Any]) -> float:
        """
        Calculate efficiency ratio for a system.
        
        E(S) = (V_A * A(S) + V_T * P(S)) / (C_R * R(S) + C_T * T(S) + Token_Cost)
        
        Args:
            system_data: Dictionary containing system performance data
                
        Returns:
            The calculated efficiency ratio
        """
        # Extract or aggregate system-level metrics if needed
        if 'model' in system_data:
            metrics = self._extract_system_metrics(system_data)
        else:
            metrics = system_data
        
        # Extract metrics
        accuracy = metrics.get('accuracy', 0.0)
        throughput = metrics.get('throughput', 0.0)
        resource_usage = metrics.get('resource_usage', 0.0) / self.config.compute_unit_normalization
        latency = metrics.get('latency', 0.0) / self.config.time_unit_normalization
        
        # Calculate value and cost components
        value = self.config.V_A * accuracy + self.config.V_T * throughput
        cost = self.config.C_R * resource_usage + self.config.C_T * latency
        
        # Avoid division by zero
        if cost <= 0:
            return float('inf') if value > 0 else 0.0
        
        return value / cost
    
    def calculate_scalability_coefficient(self, 
                                       system_data_list: List[Dict[str, Any]],
                                       problem_sizes: List[int]) -> float:
        """
        Calculate scalability coefficient for a system.
        
        SC(S) = ΔU(S) / Δn
        
        Args:
            system_data_list: List of system performance data for different problem sizes
            problem_sizes: List of problem sizes corresponding to each system_data
                
        Returns:
            The calculated scalability coefficient
        """
        if len(system_data_list) < 2 or len(problem_sizes) < 2:
            return 0.0
        
        # Sort by problem size
        paired_data = sorted(zip(problem_sizes, system_data_list), key=lambda x: x[0])
        problem_sizes, system_data_list = zip(*paired_data)
        
        # Calculate utilities for each problem size
        utilities = [self.calculate_utility(data) for data in system_data_list]
        
        # Calculate average change in utility per change in problem size
        utility_changes = [utilities[i+1] - utilities[i] for i in range(len(utilities)-1)]
        size_changes = [problem_sizes[i+1] - problem_sizes[i] for i in range(len(problem_sizes)-1)]
        
        # Calculate scalability coefficient as average of utility_change / size_change
        scalability_coefficients = [uc / sc for uc, sc in zip(utility_changes, size_changes) if sc > 0]
        
        if not scalability_coefficients:
            return 0.0
            
        return sum(scalability_coefficients) / len(scalability_coefficients)
    
    def evaluate_from_leaderboard(self, leaderboard_data_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate agent systems using data from a leaderboard file.
        
        Args:
            leaderboard_data_path: Path to the leaderboard data file (JSON or CSV)
            
        Returns:
            Dictionary mapping system names to their evaluation metrics
        """
        # Load leaderboard data
        leaderboard_data = self._load_leaderboard_data(leaderboard_data_path)
        
        # Process each system's data
        results = {}
        for system_name, system_data in leaderboard_data.items():
            # Extract system metrics if using the model-based format
            if 'model' in system_data:
                system_metrics = self._extract_system_metrics(system_data)
                # Keep the original model data in the raw_data
                system_metrics['model'] = system_data.get('model', [])
            else:
                system_metrics = system_data
            
            # Calculate evaluation metrics
            utility = self.calculate_utility(system_data)
            efficiency = self.calculate_efficiency_ratio(system_data)
            
            # Store results
            results[system_name] = {
                'raw_data': system_metrics,
                'utility': utility,
                'efficiency_ratio': efficiency,
                'system_type': system_metrics.get('system_type', 'unknown')
            }
            
        return results
    
    def evaluate_from_metrics_registry(self, metrics_registry, system_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate agent systems using data from the metrics registry.
        
        Args:
            metrics_registry: The metrics registry containing system metrics
            system_names: List of system names to evaluate
            
        Returns:
            Dictionary mapping system names to their evaluation metrics
        """
        results = {}
        
        for system_name in system_names:
            # Extract data from metrics registry
            system_data = self._extract_system_data_from_registry(metrics_registry, system_name)
            
            # Calculate evaluation metrics
            utility = self.calculate_utility(system_data)
            efficiency = self.calculate_efficiency_ratio(system_data)
            
            # Store results
            results[system_name] = {
                'raw_data': system_data,
                'utility': utility,
                'efficiency_ratio': efficiency,
                'system_type': system_data.get('system_type', 'unknown')
            }
            
        return results
    
    def _extract_system_data_from_registry(self, metrics_registry, system_name: str) -> Dict[str, Any]:
        """
        Extract system performance data from the metrics registry.
        
        Args:
            metrics_registry: The metrics registry
            system_name: Name of the system to extract data for
            
        Returns:
            Dictionary containing system performance data
        """
        system_collector = metrics_registry.get_collector("system")
        agent_collector = metrics_registry.get_collector("agent")
        
        if not system_collector or not agent_collector:
            return {}
        
        # Get accuracy from system metrics
        accuracy_metrics = system_collector.get_metrics(["system.task.evaluation.accuracy"])
        accuracy = 0.0
        if accuracy_metrics and "system.task.evaluation.accuracy" in accuracy_metrics:
            accuracy_values = [dp['value'] for dp in accuracy_metrics["system.task.evaluation.accuracy"]
                             if dp.get('tags', {}).get('system_name') == system_name]
            if accuracy_values:
                accuracy = sum(accuracy_values) / len(accuracy_values)
        
        # Calculate throughput from system metrics
        throughput = system_collector.get_throughput(task_type=f"{system_name}.task.completion")
        
        # Get latency from system metrics
        latency_metrics = system_collector.get_metrics([f"{system_name}.task.latency"])
        latency = 0.0
        if latency_metrics and f"{system_name}.task.latency" in latency_metrics:
            latency_values = [dp['value'] for dp in latency_metrics[f"{system_name}.task.latency"]]
            if latency_values:
                latency = sum(latency_values)  # Total latency across all tasks
        
        # Get resource usage from agent metrics (model parameters/computation)
         
        # Determine system type
        system_type = "single_agent"
        if "supervisor" in system_name.lower() or system_name.lower() == "supervisor_mas":
            system_type = "supervisor_mas"
        elif "swarm" in system_name.lower():
            system_type = "swarm"
        
        # Create system data dictionary
        system_data = {
            'accuracy': accuracy,
            'throughput': throughput,
            'latency': latency,
            'resource_usage': self._get_total_model_parameters(agent_collector, system_name),  # Using total model parameters as resource usage
            'system_type': system_type
        }
        
        return system_data
    
    def _get_total_model_parameters(self, agent_collector, system_name: str) -> float:
        """
        Get the total model parameters for a system from the metrics registry.
        
        Args:
            agent_collector: The agent metrics collector
            system_name: Name of the system to get parameters for
            
        Returns:
            Total model parameters for the system
        """
        # Get parameter metrics from agent collector
        parameter_metrics = agent_collector.get_metrics([f"{system_name}.model.parameters"])
        total_parameters = 0.0
        
        if parameter_metrics and f"{system_name}.model.parameters" in parameter_metrics:
            parameter_values = [dp['value'] for dp in parameter_metrics[f"{system_name}.model.parameters"]
                               if dp.get('tags', {}).get('system_name') == system_name]
            if parameter_values:
                total_parameters = sum(parameter_values)
        
        return total_parameters
    
    def _load_leaderboard_data(self, leaderboard_data_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load leaderboard data from a file.
        
        Args:
            leaderboard_data_path: Path to the leaderboard data file (JSON or CSV)
            
        Returns:
            Dictionary mapping system names to their performance data
        """
        file_ext = Path(leaderboard_data_path).suffix.lower()
        
        try:
            if file_ext == '.json':
                with open(leaderboard_data_path, 'r') as f:
                    data = json.load(f)
                    
                # Convert to expected format if necessary
                if isinstance(data, list):
                    # Convert list of systems to dictionary by system name
                    return {system['name']: system for system in data}
                return data
                
            elif file_ext == '.csv':
                df = pd.read_csv(leaderboard_data_path)
                
                # Convert DataFrame to dictionary
                if 'system_name' in df.columns:
                    return {row['system_name']: dict(row) for _, row in df.iterrows()}
                else:
                    # Assume first column is system name
                    system_name_col = df.columns[0]
                    return {row[system_name_col]: dict(row) for _, row in df.iterrows()}
                    
            else:
                print(f"Unsupported file format: {file_ext}")
                return {}
                
        except Exception as e:
            print(f"Error loading leaderboard data: {str(e)}")
            return {}
            
    def visualize_results(self, results: Dict[str, Dict[str, Any]], output_dir: Optional[str] = None) -> None:
        """
        Visualize evaluation results.
        
        Args:
            results: Dictionary mapping system names to their evaluation metrics
            output_dir: Optional directory to save visualizations
        """
        if not results:
            print("No results to visualize")
            return
            
        # Prepare data for visualization
        systems = list(results.keys())
        utilities = [results[s]['utility'] for s in systems]
        efficiencies = [results[s]['efficiency_ratio'] for s in systems]
        system_types = [results[s]['system_type'] for s in systems]
        
        # Create figure with multiple subplots
        plt.figure(figsize=(15, 15))
        
        # Plot utility values
        plt.subplot(3, 2, 1)
        sns.barplot(x=systems, y=utilities, hue=system_types)
        plt.title('Utility Values by System')
        plt.xticks(rotation=45)
        plt.ylabel('Utility Value')
        plt.tight_layout()
        
        # Plot efficiency ratios
        plt.subplot(3, 2, 2)
        sns.barplot(x=systems, y=efficiencies, hue=system_types)
        plt.title('Efficiency Ratio by System')
        plt.xticks(rotation=45)
        plt.ylabel('Efficiency Ratio')
        plt.tight_layout()
        
        # Plot accuracy vs. computational cost
        plt.subplot(3, 2, 3)
        x = [results[s]['raw_data'].get('resource_usage', 0) / self.config.compute_unit_normalization for s in systems]
        y = [results[s]['raw_data'].get('accuracy', 0) for s in systems]
        
        sns.scatterplot(x=x, y=y, hue=system_types, s=100)
        for i, system in enumerate(systems):
            plt.annotate(system, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.title('Accuracy vs. Computational Cost')
        plt.xlabel('Computational Resources (Normalized)')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        
        # Plot throughput vs. latency
        plt.subplot(3, 2, 4)
        x = [results[s]['raw_data'].get('latency', 0) / self.config.time_unit_normalization for s in systems]
        y = [results[s]['raw_data'].get('throughput', 0) for s in systems]
        
        sns.scatterplot(x=x, y=y, hue=system_types, s=100)
        for i, system in enumerate(systems):
            plt.annotate(system, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.title('Throughput vs. Latency')
        plt.xlabel('Latency (Normalized)')
        plt.ylabel('Throughput (tasks/time unit)')
        plt.tight_layout()
        
        # Plot token costs
        plt.subplot(3, 2, 5)
        token_costs = [results[s]['raw_data'].get('token_cost', 0) for s in systems]
        sns.barplot(x=systems, y=token_costs, hue=system_types)
        plt.title('Token Cost by System')
        plt.xticks(rotation=45)
        plt.ylabel('Token Cost ($)')
        plt.tight_layout()
        
        # Plot cost breakdown (resource cost vs token cost)
        plt.subplot(3, 2, 6)
        df_costs = pd.DataFrame({
            'System': systems * 2,
            'Cost Type': ['Resource Cost'] * len(systems) + ['Token Cost'] * len(systems),
            'Cost': [
                self.config.C_R * (results[s]['raw_data'].get('resource_usage', 0) / self.config.compute_unit_normalization) +
                self.config.C_T * (results[s]['raw_data'].get('latency', 0) / self.config.time_unit_normalization)
                for s in systems
            ] + [
                results[s]['raw_data'].get('token_cost', 0) for s in systems
            ]
        })
        sns.barplot(data=df_costs, x='System', y='Cost', hue='Cost Type')
        plt.title('Cost Breakdown by System')
        plt.xticks(rotation=45)
        plt.ylabel('Cost ($)')
        plt.tight_layout()
        
        # Save or display the figure
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'unified_evaluation_results.png'), dpi=300)
        else:
            plt.show()
    
    def export_results(self, results: Dict[str, Dict[str, Any]], output_path: str) -> None:
        """
        Export evaluation results to file.
        
        Args:
            results: Dictionary mapping system names to performance metrics
            output_path: Path to export results to
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
    def evaluate_inference_metrics(self, result_files: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate inference metrics from benchmark result files.
        
        For open-source models (estimable via SystemMetricsCollector), TTFT, throughput (tokens/sec),
        and memory usage are derived from estimations.
        For closed-source models, observed latency is used, and estimated memory/throughput are not applicable from these functions.
        
        Args:
            result_files: List of paths to benchmark result files
            
        Returns:
            Dictionary mapping agent system names to metrics including estimated values for open models.
        """
        results = {}
        system_metrics_collector = SystemMetricsCollector() # Instantiate once
        
        for file_path in result_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            agent_system_name = parts[1]

            if not data:
                results[agent_system_name] = {
                    'accuracy': 0,
                    'throughput_tasks_per_sec': 0,
                    'latency_ttft_ms': 0,
                    'estimated_memory_mb': 0,
                    'throughput_estimated_tokens_per_sec': 0,
                    'model': [],
                    'error': 'No data in result file'
                }
                continue

            total_problems = len(data)
            correct = sum(1 for item in data if item.get('score', 0) == 1)
            accuracy = correct / total_problems if total_problems > 0 else 0
            
            total_duration_ms_observed = sum(item.get('duration_ms', 0) for item in data)
            avg_problem_duration_ms = total_duration_ms_observed / total_problems if total_problems > 0 else 0
            throughput_tasks_per_sec = 1000.0 / avg_problem_duration_ms if avg_problem_duration_ms > 0 else 0

            all_agent_calls_details = []
            
            for item in data:
                llm_usage = item.get('llm_usage', {})
                agent_usage = llm_usage.get('agent_usage', [])
                
                for agent_call in agent_usage:
                    model_name = agent_call.get('model_name', 'unknown_model')
                    prompt_tokens = agent_call.get('prompt_tokens', 0)
                    completion_tokens = agent_call.get('completion_tokens', 0)
                    observed_latency_ms = agent_call.get('latency_ms', 0)
                    
                    estimated_metrics = system_metrics_collector.estimate_inference_metrics(
                        model_name=model_name,
                        input_token_count=prompt_tokens,
                        output_token_count=completion_tokens
                    )
                    
                    all_agent_calls_details.append({
                        'model_name': model_name,
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'observed_latency_ms': observed_latency_ms,
                        'estimated_metrics': estimated_metrics # Store the whole dict or None
                    })

            # Aggregate metrics
            sum_latency_ttft_contributions_ms = 0
            sum_estimated_memory_bytes = 0
            sum_estimated_output_tokens = 0
            sum_estimated_generation_time_seconds = 0
            num_agent_invocations = len(all_agent_calls_details)

            for call_detail in all_agent_calls_details:
                if call_detail['estimated_metrics']: # Open-source / estimable
                    sum_latency_ttft_contributions_ms += call_detail['estimated_metrics'].get('ttft_seconds', 0) * 1000
                    sum_estimated_memory_bytes += call_detail['estimated_metrics'].get('memory_usage_bytes', 0)
                    
                    tokens_per_sec = call_detail['estimated_metrics'].get('tokens_per_second', 0)
                    if tokens_per_sec > 0:
                        sum_estimated_output_tokens += call_detail['completion_tokens']
                        sum_estimated_generation_time_seconds += call_detail['completion_tokens'] / tokens_per_sec
                else: # Closed-source / not estimable
                    sum_latency_ttft_contributions_ms += call_detail['observed_latency_ms']

            avg_overall_latency_ttft_ms = sum_latency_ttft_contributions_ms / num_agent_invocations if num_agent_invocations > 0 else 0
            overall_estimated_memory_mb = sum_estimated_memory_bytes / (1024 * 1024)
            
            overall_throughput_estimated_tokens_per_sec = 0
            if sum_estimated_generation_time_seconds > 0:
                overall_throughput_estimated_tokens_per_sec = sum_estimated_output_tokens / sum_estimated_generation_time_seconds

            # Process unique model information for the 'model' list
            final_models_info = []
            if all_agent_calls_details:
                unique_model_names = sorted(list(set(cd['model_name'] for cd in all_agent_calls_details)))
                
                for uname in unique_model_names:
                    calls_for_this_model = [cd for cd in all_agent_calls_details if cd['model_name'] == uname]
                    
                    model_info_entry = {
                        'model_name': uname,
                        'invocations': len(calls_for_this_model),
                        'total_prompt_tokens': sum(cd['prompt_tokens'] for cd in calls_for_this_model),
                        'total_completion_tokens': sum(cd['completion_tokens'] for cd in calls_for_this_model)
                        }
            
                    # Check if this model type is estimable (based on the first call, assuming consistency)
                    first_call_estimable = calls_for_this_model[0]['estimated_metrics'] is not None

                    if first_call_estimable:
                        estimable_ttfts_ms = [cd['estimated_metrics']['ttft_seconds'] * 1000 for cd in calls_for_this_model if cd['estimated_metrics']]
                        model_info_entry['latency_ttft_ms'] = sum(estimable_ttfts_ms) / len(estimable_ttfts_ms) if estimable_ttfts_ms else 0
                        
                        estimable_tps = [cd['estimated_metrics']['tokens_per_second'] for cd in calls_for_this_model if cd['estimated_metrics'] and cd['estimated_metrics'].get('tokens_per_second', 0) > 0]
                        model_info_entry['tokens_per_second'] = sum(estimable_tps) / len(estimable_tps) if estimable_tps else 0
                        
                        estimable_memory_bytes = sum(cd['estimated_metrics']['memory_usage_bytes'] for cd in calls_for_this_model if cd['estimated_metrics'])
                        model_info_entry['estimated_memory_mb'] = estimable_memory_bytes / (1024 * 1024)
                    else:
                        observed_latencies = [cd['observed_latency_ms'] for cd in calls_for_this_model]
                        model_info_entry['latency_ttft_ms'] = sum(observed_latencies) / len(observed_latencies) if observed_latencies else 0
                    
                    final_models_info.append(model_info_entry)
            
            results[agent_system_name] = {
                'accuracy': accuracy,
                'throughput_tasks_per_sec': throughput_tasks_per_sec,
                'latency_ttft_ms': avg_overall_latency_ttft_ms,
                'estimated_memory_mb': overall_estimated_memory_mb, # Memory contributed by estimable (open-source) models
                'throughput_estimated_tokens_per_sec': overall_throughput_estimated_tokens_per_sec, # Throughput for estimable models
                'model': final_models_info
            }
        
        return results


class LeaderboardInstrumentationCollector(BaseMetricsCollector):
    """
    Collector that uses leaderboard data as instrumentation for evaluation.
    
    This collector transforms leaderboard data into metrics that can be
    used with the unified evaluation framework.
    """
    
    def __init__(self, config: Optional[MetricsCollectionConfig] = None):
        """
        Initialize the leaderboard instrumentation collector.
        
        Args:
            config: Configuration for metrics collection
        """
        super().__init__(config or MetricsCollectionConfig())
        self.leaderboard_data = {}
        
    def load_leaderboard_data(self, data_path: str) -> None:
        """
        Load leaderboard data from a file.
        
        Args:
            data_path: Path to the leaderboard data file
        """
        try:
            # Load data based on file extension
            file_ext = Path(data_path).suffix.lower()
            
            if file_ext == '.json':
                with open(data_path, 'r') as f:
                    self.leaderboard_data = json.load(f)
            elif file_ext == '.csv':
                df = pd.read_csv(data_path)
                self.leaderboard_data = df.to_dict(orient='records')
            else:
                print(f"Unsupported file format: {file_ext}")
                return
                
            # Convert data to metrics
            self._convert_leaderboard_to_metrics()
            
        except Exception as e:
            print(f"Error loading leaderboard data: {str(e)}")
    
    def _convert_leaderboard_to_metrics(self) -> None:
        """Convert leaderboard data into metrics in the standard format."""
        if not self.leaderboard_data:
            return
            
        # Process each system in the leaderboard
        for system in self.leaderboard_data:
            system_name = system.get('name', 'unknown_system')
            
            # Record accuracy
            if 'accuracy' in system:
                self.collect_point(
                    metric_name="system.task.evaluation.accuracy",
                    value=system['accuracy'],
                    tags={'system_name': system_name}
                )
            
            # Record throughput
            if 'throughput' in system:
                self.collect_point(
                    metric_name=f"{system_name}.task.throughput",
                    value=system['throughput'],
                    tags={'system_name': system_name, 'unit': 'tasks/time_unit'}
                )
            
            # Record latency
            if 'latency' in system:
                self.collect_point(
                    metric_name=f"{system_name}.task.latency",
                    value=system['latency'],
                    tags={'system_name': system_name, 'unit': 'ms'}
                )
            
            # Record resource usage
            if 'parameters' in system:
                self.collect_point(
                    metric_name=f"{system_name}.resource.parameters",
                    value=system['parameters'],
                    tags={'system_name': system_name, 'unit': 'count'}
                )
            
            # Record activated parameters (computation)
            if 'activated_parameters' in system:
                self.collect_point(
                    metric_name=f"{system_name}.resource.activated_parameters",
                    value=system['activated_parameters'],
                    tags={'system_name': system_name, 'unit': 'count'}
                ) 