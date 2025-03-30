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
│   │   └── supervisor_mas.py
│   ├── data/              # Benchmark datasets
│   │   ├── math_test.jsonl
│   │   ├── drop_test.jsonl
│   │   └── ...
│   ├── evaluators/        # Task-specific evaluation modules
│   │   ├── math.py
│   │   ├── benchmark.py
│   │   └── ...
│   ├── workload/          # Workload generation modules
│   │   ├── generator.py
│   │   └── ...
│   ├── instrumentation/   # System monitoring tools
│   │   ├── graph_instrumentation.py
│   │   └── ...
│   ├── metrics/           # Performance metrics collection
│   │   ├── system_metrics.py
│   │   ├── agent_metrics.py
│   │   ├── inter_agent_metrics.py
│   │   ├── collectors.py
│   │   └── ...
│   ├── analysis/          # Analysis of benchmark results
│   │   └── ...
│   ├── visualization/     # Visualization of benchmark results
│   │   └── ...
│   └── utils/             # Utility functions
│       └── ...
├── benchmark_runner.py    # Simplified interface for running benchmarks
├── run_benchmark.py       # Main entry point for running benchmarks
└── README.md              # This file
```

## Core Components

1. **Agents**: Implementations of single and multi-agent systems, including supervisor-based approaches.
2. **Evaluators**: Modules for evaluating agent performance on specific tasks.
3. **Data**: Benchmark datasets for various tasks (MATH, DROP, HotpotQA, etc.).
4. **Workload**: Generation of test scenarios and agent tasks.
5. **Instrumentation**: Non-intrusive mechanisms to monitor system behavior.
6. **Metrics**: Collection and processing of performance data.
7. **Analysis**: Tools for identifying bottlenecks and optimization opportunities.
8. **Visualization**: Dashboards and graphs for result visualization.

## Getting Started

### Prerequisites

- Python 3.9+
- Install requirements:
  ```
  pip install -r requirements.txt
  ```

### Running a Benchmark

#### Simple Interface (Recommended)

The easiest way to run a benchmark is using the simplified interface:

```python
# One-line benchmark
from benchmark.benchmark_runner import run_simple_benchmark
summary = run_simple_benchmark(benchmark_name="math", limit=5)

# Or using the Benchmark class for more control
from benchmark.benchmark_runner import Benchmark
benchmark = Benchmark()
summary = benchmark.run("math", limit=5)
benchmark.visualize_results()
```

You can also run from the command line:

```bash
python benchmark_runner.py --benchmark math --limit 5
```

This will:
1. Run the benchmark on the specified dataset
2. Collect detailed metrics automatically
3. Save results and metrics to disk
4. Generate visualizations and summary reports
5. Return a summary with accuracy and performance stats

#### Advanced Interface

For more control, you can use the original interface:

```bash
python run_benchmark.py --benchmark math --limit 10 --metrics-output metrics
```

Options:
- `--benchmark`: Choose the benchmark to run (math, drop, gsm8k, hotpotqa, humaneval, mbpp)
- `--data-path`: Custom path to benchmark data file
- `--limit`: Limit the number of problems to process
- `--output-dir`: Directory to save results
- `--metrics-output`: Directory to save metrics data

### Token Usage Tracking

The benchmark framework tracks token usage by agent to measure the efficiency of different multi-agent architectures:

1. **Per-Agent Tracking**: Token usage is tracked separately for each agent (e.g., supervisor, researcher, coder)
2. **Total Tokens**: The framework records total tokens used by each agent (not split between prompt/completion)
3. **Metrics Collection**: Token usage is stored in metrics and available for analysis
4. **Efficiency Analysis**: The framework can calculate metrics like tokens-per-correct-answer

Example token tracking:

```python
from benchmark.benchmark_runner import Benchmark

benchmark = Benchmark()
summary = benchmark.run("math", limit=3)

# Token usage is available in the results
for result in benchmark.results:
    for agent_id, tokens in result.get("token_usage", {}).items():
        print(f"{agent_id}: {tokens} tokens")
```

See `examples/token_tracking_example.py` for a complete example of token tracking analysis.

### Metrics Collection and Analysis

The framework includes comprehensive metrics collection for monitoring and analyzing agent system performance:

1. **System Metrics**: Overall throughput, latency, and resource utilization.
2. **Agent Metrics**: LLM usage, token consumption, and agent-specific performance.
3. **Inter-Agent Metrics**: Communication patterns and coordination overhead.

#### Example: Using the Metrics Framework

```python
from benchmark.src.metrics import (
    MetricsRegistry, 
    SystemMetricsCollector,
    AgentMetricsCollector
)

# Initialize registry and collectors
registry = MetricsRegistry()
registry.register_collector("system", SystemMetricsCollector())
registry.register_collector("agent", AgentMetricsCollector())

# Start collecting metrics
registry.start_all_collectors()

# Run your agent system with metrics collection
results = evaluate_mas(problem, metrics_registry=registry)

# Export and analyze metrics
registry.export_all("json", "metrics_output")
registry.stop_all_collectors()
```

For a complete example, see `examples/metrics_collection_example.py`.

## Extending the Framework

### Adding a New Agent System

1. Create a new file in `src/agents/`
2. Implement your agent system
3. Create an evaluation function compatible with the benchmark framework

### Adding a New Benchmark Dataset

1. Add your dataset to `src/data/`
2. Create a corresponding evaluator in `src/evaluators/`
3. Update the main runner to support your new benchmark

### Creating Custom Metrics Collectors

1. Extend the `BaseMetricsCollector` class in `src/metrics/collectors.py`
2. Implement your custom metrics collection logic
3. Register your collector with the `MetricsRegistry`

## License

This project is licensed under the MIT License - see the LICENSE file for details. 