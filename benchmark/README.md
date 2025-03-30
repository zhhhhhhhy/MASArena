# Multi-Agent System Benchmark

A comprehensive framework for benchmarking single and multi-agent systems across various tasks, evaluating their performance, accuracy, and efficiency.

## Project Structure

```
benchmark/
├── examples/              # Example code demonstrating framework usage
│   ├── metrics_usage_example.py
│   ├── metrics_collection_example.py
│   ├── token_tracking_example.py
│   └── langgraph_integration.py
├── src/                   # Core source code
│   ├── agents/            # Agent system implementations
│   │   ├── base.py        # Base classes for agent systems
│   │   ├── single_agent.py # Single agent implementation
│   │   ├── supervisor_mas.py # Supervisor-based multi-agent system
│   │   ├── swarm.py       # Swarm-based multi-agent system
│   │   └── __init__.py    # Registry for agent systems
│   ├── data/              # Benchmark datasets
│   │   ├── math_test.jsonl # Mathematics problems
│   │   ├── ....
│   ├── evaluators/        # Task-specific evaluation modules
│   │   ├── base.py        # Base evaluator class
│   │   ├── math.py        # Mathematics evaluator
│   │   ├── ....
│   │   └── benchmark.py   # Common evaluation utilities
│   ├── instrumentation/   # System monitoring tools
│   │   ├── graph_instrumentation.py
│   │   └── ...
│   ├── metrics/           # Performance metrics collection
│   │   ├── system_metrics.py
│   │   ├── agent_metrics.py
│   │   ├── inter_agent_metrics.py
│   │   ├── collectors.py
│   │   └── ...
│       └── ...
├── benchmark_runner.py    # Simplified interface for running benchmarks
└── README.md              
```

## Core Components

1. **Agents**: Implementations of single and multi-agent systems, including:
   - **Single Agent**: Simple agent that uses a single LLM to solve problems directly
   - **Supervisor-based MAS**: Coordinator agent that directs specialized worker agents
   - **Swarm Agent**: Multiple independent agents solving the same problem with aggregation

2. **Evaluators**: Modules for evaluating agent performance on specific tasks:
   - **Math**: Evaluates mathematical problem solving with exact answer matching
   - ...

3. **Data**: Benchmark datasets for various tasks

4. **Instrumentation**: Non-intrusive mechanisms to monitor system behavior.
5. **Metrics**: Collection and processing of performance data.
