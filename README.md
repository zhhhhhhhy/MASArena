# Multi-Agent Benchmark

A comprehensive framework for benchmarking single and multi-agent systems across various tasks, evaluating their performance, accuracy, and efficiency.

## Quick Start

### Setup

We highly recommend using [uv](https://docs.astral.sh/uv/) to manage project dependencies:

```bash
# Install dependencies
uv sync

# Setup pre-commit hooks
pre-commit install
```

When you want to add a dependency to the project:

```bash
uv add [package]
```

The packages installed by `pip` would NOT be added into project dependencies.

### Running Benchmarks

The main way to run benchmarks is through the `main.py` script:

```bash
# Run a math benchmark with the single agent system
python main.py --benchmark math --agent-system single_agent --limit 5

# Run a benchmark with the supervisor-based multi-agent system
python main.py --benchmark math --agent-system supervisor_mas --limit 10

# Run with swarm-based agent system
python main.py --benchmark math --agent-system swarm --limit 5
```

You can also use the runner scripts for more options:

```bash
# Using the bash runner
./run_benchmark.sh math swarm 5

# Using the Python runner with all agent systems
./run_benchmark.py --benchmark math --all-agents --limit 5
```



### Visualizing Agent Interactions

The framework provides interactive visualizations for agent interactions and benchmark results:

```bash
python benchmark/src/visualization/visualize_benchmark.py visualize --summary results/math_swarm_20250423_170316_summary.json
```

## Available Agent Systems

1. **Single Agent**: A single LLM solving problems directly
2. **Supervisor MAS**: A supervisor coordinating specialized worker agents
3. **Swarm**: Multiple independent agents solving the same problem with aggregation

## Available Benchmarks

- **math**: Mathematical problems of varying difficulty
- **drop**: Reading comprehension with numerical reasoning
- **gsm8k**: Grade school math word problems
- **hotpotqa**: Multi-hop question answering
- **humaneval**: Code generation problems
- **mbpp**: Python programming problems

## Project Structure

```
project_multi_agents_benchmark/
├── benchmark/              # Benchmark framework
│   ├── data/               # Benchmark datasets
│   ├── src/                # Core source code
│   │   ├── agents/         # Agent system implementations
│   │   ├── evaluators/     # Task-specific evaluation modules
│   │   ├── metrics/        # Performance metrics collection
│   │   └── instrumentation/ # Instrumentation for system metrics
│   └── benchmark_runner.py # Simplified interface for running benchmarks
│   └── README.md           # Documentation for the benchmark
├── main.py                 # Main entry point for running benchmarks
├── run_benchmark.sh        # Bash runner script
├── analyze_results.py      # Results analysis tool
├── results/                # Benchmark results for token usage & performance metrics
├── README.md               # Documentation for the project
```


## Logic of the benchmark
Agent System → Instrumentation → Metrics
(What to test) → (How to monitor) → (What to measure)

1. Agent System generates realistic test scenarios and agent tasks
2. Instrumentation captures system behavior and events
3. Metrics processes and structures the captured data


### Agent System
The Agent System module serves as the foundation for generating realistic and reproducible testing scenarios for multi-agent systems. 

### Instrumentation
The Instrumentation module provides non-intrusive mechanisms to monitor and track multi-agent system behavior without significantly altering performance characteristics. It serves as the sensing layer of the benchmark framework, capturing relevant events and measurements across different components.

* **Graph Instrumentation**: Specialized instrumentation for **graph-based agent workflows** like LangGraph, tracking message passing, node execution, and graph traversal patterns. Captures the structural dynamics of agent interactions.
* **LLM Instrumentation**: Monitors LLM-related operations, including prompt construction, token usage, and response processing. Provides hooks for timing API calls and capturing cost metrics while preserving the context of agent operations.
* **Tool Instrumentation**: Tracks agent tool usage, including invocation patterns, parameter distributions, and execution outcomes. Enables analysis of tool effectiveness and agent decision-making.
* **Memory Instrumentation**: Monitors memory operations across agent systems, tracking retrieval patterns, storage efficiency, and memory growth over time. Essential for understanding the long-term performance of agents with memory components.
The Instrumentation module uses strategic interceptors and wrappers that maintain the semantic equivalence of instrumented and non-instrumented code, ensuring that measurements accurately reflect natural system behavior while providing comprehensive visibility into operations.

### Metrics
The Metrics module defines the data collection, processing, and storage infrastructure for capturing multi-agent system performance data. It establishes a uniform metrics model that enables cross-system comparisons while accommodating domain-specific measurements.

* **System Metrics**: Collects system-level performance indicators including throughput, latency distributions, resource utilization, and overall system stability. Provides a macroscopic view of multi-agent performance.
* **Agent Metrics**: Focuses on individual agent performance metrics such as token usage, tool utilization patterns, decision-making efficiency, and memory operations. Enables fine-grained analysis of agent behavior and optimization opportunities.
* **Inter-Agent Metrics**: Captures communication patterns, coordination overhead, and interaction dynamics between agents. Essential for understanding emergent behaviors in multi-agent systems and identifying collaboration bottlenecks.
* **Collectors**: Implements the core collection infrastructure with efficient, thread-safe mechanisms for gathering, buffering, and processing metrics. Includes configurable sampling strategies to balance measurement accuracy and overhead.
The Metrics module employs queue-based processing with configurable batch sizes and sampling rates to efficiently handle high volumes of metrics data while minimizing impact on the measured system. 

## Command Line Options

The `main.py` script supports the following options:

```
--benchmark BENCHMARK     Benchmark to run (default: math)
                          Choices: math, drop, gsm8k, hotpotqa, humaneval, mbpp
--agent-system SYSTEM     Agent system to use (default: single_agent)
                          Choices: single_agent, supervisor_mas, swarm
--limit LIMIT             Maximum number of problems to process (default: 10)
--data PATH               Path to benchmark data (optional)
--results-dir DIR         Directory to store results (default: results)
--metrics-dir DIR         Directory to store metrics (default: metrics)
--verbose                 Print progress information (default: True)
```

## Development

### Code Style

For code formatting and linting:

```bash
ruff check
ruff format
```

### Agent System Format

The agent system format is a JSON object to be evaluated by the UnifiedEvaluator.
```
{
  "agent_system_name": {
    "system_type": "single_agent"|"mas"|"swarm",
    "accuracy": float,  // Performance accuracy (0.0-1.0)
    "throughput": float, // number of tasks per second
    "latency": float, // ttft on average for all tasks
    "model": [
      {
        "model_name": string,  // Name of the LLM used
        "latency": int,        // ttft
        "input_token_count": int,  // Number of input tokens
        "output_token_count": int, // Number of output tokens
      },
      // Additional models for multi-agent systems
    ]
  },
  // Additional agent systems
}
```

### Extending the Framework

#### Adding a New Agent System

1. Create a new file in `benchmark/src/agents/` (e.g., `my_agent.py`)
2. Implement your agent system by subclassing `AgentSystem` from `base.py`
3. Register your system with the `AgentSystemRegistry`
4. Import your agent system in `__init__.py` to make it available

#### Adding a New Benchmark Dataset

1. Add your dataset to `results/`
2. Create a corresponding evaluator in `benchmark/src/evaluators/`
3. Update the available choices in `main.py`
