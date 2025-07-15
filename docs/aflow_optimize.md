# AFlowOptimizer User Guide

## Introduction

AFlowOptimizer is a core component of the MASArena framework for automated optimization of multi-agent workflows. It leverages LLM-driven evolutionary optimization to automatically modify and evaluate workflow code (such as `graph.py` and `prompt.py`), aiming to improve performance on a specified benchmark.

AFlow supports multi-round iterative optimization. In each round, it generates new workflow variants based on historical performance, validates them on evaluation sets, and selects the best-performing solution.

---

## Key Features

- **Automated Evolutionary Optimization**: Uses LLM feedback to automatically modify workflow structure and prompts.
- **Multi-round Iteration**: Supports multiple optimization rounds and convergence checks.
- **Benchmark Agnostic**: Works with various benchmarks (e.g., humaneval, math).
- **Highly Extensible**: Supports custom operators, agents, and evaluators.

---

## Quick Start

### 1. Environment Setup

- Ensure you have set the following environment variables (e.g., in a `.env` file):
  - `OPENAI_API_KEY`
  - `OPENAI_API_BASE`
  - (Optional) `OPTIMIZER_MODEL_NAME`, `EXECUTOR_MODEL_NAME`

### 2. Run Optimization

Use the provided script:

```bash
chmod +x example/aflow/run_aflow_optimize.sh
./example/aflow/run_aflow_optimize.sh
```

- The script will optimize the workflow for the selected benchmark and save results to the specified path.
- You can also pass arguments to the script to override default values:
  ```bash
  ./example/aflow/run_aflow_optimize.sh humaneval mas_arena/configs/aflow example/aflow/humaneval/optimization 1 1 3
  ```

---

## Script Arguments

| Argument             | Type   | Default                                      | Description                                      |
|----------------------|--------|----------------------------------------------|--------------------------------------------------|
| `--benchmark`        | str    | humaneval                                    | Benchmark to run. Currently, only `humaneval` is supported by this script. |
| `--graph_path`       | str    | mas_arena/configs/aflow                      | Path to the AFlow graph configuration.            |
| `--optimized_path`   | str    | example/aflow/humaneval/optimization         | Path to save the optimized AFlow graph.           |
| `--validation_rounds`| int    | 1                                            | Number of validation rounds per optimization.     |
| `--eval_rounds`      | int    | 1                                            | Number of evaluation rounds per optimization.     |
| `--max_rounds`       | int    | 3                                            | Maximum number of optimization rounds.            |

---

## Example Usage

```python
import os
from dotenv import load_dotenv
from mas_arena.agents import AgentSystemRegistry
from mas_arena.evaluators import BENCHMARKS
from mas_arena.optimizers.aflow.aflow_optimizer import AFlowOptimizer
from mas_arena.optimizers.aflow.aflow_experimental_config import EXPERIMENTAL_CONFIG

# Load environment variables
load_dotenv()

# --- Configuration ---
BENCHMARK_NAME = "humaneval"
GRAPH_PATH = "mas_arena/configs/aflow"
OPTIMIZED_PATH = f"example/aflow/{BENCHMARK_NAME}/optimization"

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL_NAME", "gpt-4o")
EXECUTOR_MODEL = os.getenv("EXECUTOR_MODEL_NAME", "gpt-4o-mini")

# --- Initialization ---
optimizer_agent = AgentSystemRegistry.get(
    "single_agent", {"model_name": OPTIMIZER_MODEL, "api_key": API_KEY, "api_base": API_BASE}
)
executor_agent = AgentSystemRegistry.get(
    "single_agent", {"model_name": EXECUTOR_MODEL, "api_key": API_KEY, "api_base": API_BASE}
)

# Dynamically load evaluator based on benchmark
evaluator_class = BENCHMARKS[BENCHMARK_NAME]["evaluator"]
evaluator = evaluator_class(BENCHMARK_NAME, {})


# --- Optimizer Setup ---
optimizer = AFlowOptimizer(
    graph_path=GRAPH_PATH,
    optimized_path=OPTIMIZED_PATH,
    optimizer_agent=optimizer_agent,
    executor_agent=executor_agent,
    validation_rounds=1,
    eval_rounds=1,
    max_rounds=3,
    **EXPERIMENTAL_CONFIG.get(BENCHMARK_NAME, {})
)

# --- Run Optimization ---
optimizer.setup()
optimizer.optimize(evaluator)
optimizer.test(evaluator)
```

---

## Workflow

1. **Setup**: Loads agents, evaluator, and configuration.
2. **Optimization**: Iteratively generates and evaluates workflow variants.
3. **Validation**: Runs validation rounds to select the best variant.
4. **Testing**: Tests the final workflow and saves results.

---

## Configuration

- **Graph Path**: Points to the base workflow configuration (e.g., `mas_arena/configs/aflow`).
- **Optimized Path**: Where new workflow variants and results are saved (e.g., `example/aflow/humaneval/optimization`).
- **EXPERIMENTAL_CONFIG**: Contains benchmark-specific settings (see `mas_arena/optimizers/aflow/aflow_experimental_config.py`).

---

## FAQ

**Q: What models are used for optimization and execution?**
A: By default, `gpt-4o` for optimization and `gpt-4o-mini` for execution. You can override via environment variables.

**Q: How do I add a new benchmark?**
A: Implement a new evaluator in `mas_arena/evaluators/`, register it in `BENCHMARKS`, and provide a config in `EXPERIMENTAL_CONFIG`.

**Q: Where are the optimized workflows saved?**
A: In the directory specified by `--optimized_path` (default: `example/aflow/humaneval/optimization`).

---

## References
- See `example/aflow/run_aflow_optimize.py` for the latest usage pattern.
- See `mas_arena/optimizers/aflow/aflow_optimizer.py` for optimizer implementation.
- See `mas_arena/optimizers/aflow/aflow_experimental_config.py` for configuration details.
