# Multi-Agent System-level Evaluation Framework

This module provides a comprehensive framework for collecting, analyzing, and visualizing performance metrics for multi-agent systems built with LangGraph.

## Overview

The metrics framework enables detailed monitoring of:

1. **System-level performance** - Throughput, latency, resource utilization
2. **Agent-level behavior** - LLM usage, tool performance, memory operations  
3. **Inter-agent dynamics** - Communication patterns, coordination overhead, error propagation

## Installation

Ensure you have the required dependencies:

```bash
pip install psutil numpy matplotlib pandas networkx
# Optional for GPU monitoring
pip install gputil
```

## Quick Start

Here's a minimal example to get started:

```python
from benchmark.src.metrics import (
    MetricsRegistry, 
    SystemMetricsCollector,
    AgentMetricsCollector
)

# Initialize registry and collectors
registry = MetricsRegistry()

# System metrics
system_collector = SystemMetricsCollector()
registry.register_collector("system", system_collector)

# Agent metrics
agent_collector = AgentMetricsCollector()
registry.register_collector("agent", agent_collector)

# Start collecting metrics
registry.start_all_collectors()

# Record some metrics during your application run
system_collector.record_latency("operation_name", 120.5)  # latency in ms
agent_collector.record_llm_usage(
    agent_id="researcher", 
    model_name="gpt-4",
    prompt_tokens=1024, 
    completion_tokens=512,
    latency_ms=450.0
)

# When finished, export metrics
registry.export_all("json", "metrics_output")
registry.stop_all_collectors()
```

## Integration with LangGraph

To instrument a LangGraph-based multi-agent system:

1. Initialize metrics collectors in your application
2. Instrument LangGraph nodes with metrics collection
3. Analyze and visualize the metrics

Example:

```python
# Create an instrumented agent node
def create_instrumented_agent(agent_id, llm):
    registry = MetricsRegistry()
    agent_collector = registry.get_collector("agent")
    
    def agent_node(state):
        # Track start time
        start_time = time.time()
        
        # Your agent logic here
        result = llm.invoke(state["input"])
        
        # Record metrics
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        agent_collector.record_llm_usage(
            agent_id=agent_id,
            model_name=llm.model_name,
            prompt_tokens=len(state["input"]) // 4,  # estimated
            completion_tokens=len(result) // 4,  # estimated
            latency_ms=latency_ms
        )
        
        return {"output": result}
    
    return agent_node

# Build your graph with instrumented nodes
workflow = StateGraph()
workflow.add_node("agent_a", create_instrumented_agent("agent_a", llm_a))
workflow.add_node("agent_b", create_instrumented_agent("agent_b", llm_b))
# ... add edges and compile ...
```

## Visualization

The metrics framework includes functions for basic analysis and visualization:

```python
import matplotlib.pyplot as plt
from benchmark.src.metrics import MetricsRegistry

# Get collectors
registry = MetricsRegistry()
system_collector = registry.get_collector("system")

# Get latency percentiles
percentiles = system_collector.get_latency_percentiles("operation_name")
print(f"p50: {percentiles[50.0]} ms, p99: {percentiles[99.0]} ms")

# Plot throughput over time
metrics = system_collector.get_metrics(["system.task.completion.agent_response"])
timestamps = [datetime.fromisoformat(dp['timestamp']) for dp in metrics]
values = [dp['value'] for dp in metrics]
plt.plot(timestamps, values)
plt.title("Agent Response Throughput")
plt.show()
```

## Advanced Configuration

Each collector can be configured with specific settings:

```python
from benchmark.src.metrics import SystemMetricsConfig, SystemMetricsCollector

config = SystemMetricsConfig(
    sampling_interval_ms=5000,  # Sample every 5 seconds
    monitor_gpu=True,           # Enable GPU monitoring
    throughput_window_seconds=60,  # Calculate throughput over 1 minute
    latency_percentiles=[50.0, 90.0, 99.0, 99.9],  # Specific percentiles
    process_ids_to_monitor={1234, 5678}  # Monitor specific processes
)

collector = SystemMetricsCollector(config)
```

## Inference Metrics Evaluation

The `UnifiedEvaluator` class provides functionality to evaluate benchmark results and generate detailed inference metrics for agent systems. This is particularly useful for comparing different agent architectures or configurations.

```python
from benchmark.src.metrics.unified_evaluator import UnifiedEvaluator

# Initialize the evaluator
evaluator = UnifiedEvaluator()

# Evaluate benchmark results from multiple systems
results = evaluator.evaluate_inference_metrics([
    "results/math_swarm_20250413_113622.json",
    "results/math_agentverse_20250413_141244.json"
])

# Example of the results structure:
"""
{
  "agentverse": {
    "accuracy": 0.66666,
    "throughput": 0.07972,         # Tasks per second
    "latency": 348.36,             # Average TTFT in milliseconds
    "memory": 156211.92,           # Total estimated memory in MB
    "model": [
      {
        "model_name": "gpt-4o-mini",
        "latency": 212.10,         # TTFT in milliseconds
        "input_token_count": 293,
        "output_token_count": 259
      }
    ]
  },
  "swarm": {
    "accuracy": 0.5,
    "throughput": 0.034,
    "latency": 298.12,
    "memory": 184325.73,
    "model": [
      {
        "model_name": "gpt-4o-mini",
        "latency": 212.10,
        "input_token_count": 320,
        "output_token_count": 480
      }
    ]
  }
}
"""

# Example usage with BenchmarkRunner
from benchmark.benchmark_runner import BenchmarkRunner

benchmark = BenchmarkRunner()
summary = benchmark.run("math", limit=5, agent_system="swarm")
# The summary includes inference metrics automatically:
print(summary["inference_metrics"]["accuracy"])  # 0.5
```

### Data Structure Details

The inference metrics evaluation provides the following information:

- **accuracy**: Fraction of problems solved correctly (0.0-1.0)
- **throughput**: Number of tasks the system can process per second
- **latency**: Average time to first token (TTFT) across all requests in milliseconds
- **memory**: Total estimated memory usage in megabytes (includes model parameters, KV cache, and activations)
- **model**: List of unique models used by the agent system, with per-model metrics:
  - **model_name**: Name of the language model
  - **latency**: Estimated time to first token in milliseconds
  - **input_token_count**: Average number of prompt tokens
  - **output_token_count**: Average number of completion tokens

The `estimate_inference_metrics` function from `SystemMetricsCollector` is used internally to calculate these metrics based on model parameters, input token count, and output token count. This provides a standardized way to compare different agent architectures even when direct measurements are not available.

## Custom Metrics

You can extend the framework with custom metrics:

```python
# Record custom metrics
system_collector.collect_point(
    metric_name="custom.metric.name",
    value=42.0,
    tags={"dimension1": "value1", "dimension2": "value2"}
)
```

## Examples

See [langgraph_integration.py](../../examples/langgraph_integration.py) for a complete example of integrating the metrics framework with a LangGraph-based multi-agent system. 