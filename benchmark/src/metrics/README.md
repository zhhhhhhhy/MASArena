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