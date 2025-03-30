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

### Analyzing Results

After running benchmarks, analyze the results with:

```bash
# Compare performance across all agent systems
./analyze_results.py --compare

# Analyze a specific results file
./analyze_results.py results/math_swarm_20240330_123456.json
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
│   ├── src/                # Core source code
│   │   ├── agents/         # Agent system implementations
│   │   ├── data/           # Benchmark datasets
│   │   ├── evaluators/     # Task-specific evaluation modules
│   │   ├── metrics/        # Performance metrics collection
│   │   └── ...
│   └── benchmark_runner.py # Simplified interface for running benchmarks
├── main.py                 # Main entry point for running benchmarks
├── run_benchmark.sh        # Bash runner script
├── analyze_results.py      # Results analysis tool
├── results/                # Benchmark results for token usage & performance metrics
└── metrics/                # Collected metrics data for system metrics
```

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

### Extending the Framework

#### Adding a New Agent System

1. Create a new file in `benchmark/src/agents/` (e.g., `my_agent.py`)
2. Implement your agent system by subclassing `AgentSystem` from `base.py`
3. Register your system with the `AgentSystemRegistry`
4. Import your agent system in `__init__.py` to make it available

#### Adding a New Benchmark Dataset

1. Add your dataset to `benchmark/src/data/`
2. Create a corresponding evaluator in `benchmark/src/evaluators/`
3. Update the available choices in `main.py`
