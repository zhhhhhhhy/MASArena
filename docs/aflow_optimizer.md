# AFlowOptimizer User Guide

## Introduction

AFlowOptimizer is a core component of the MASArena framework for automated optimization of multi-agent workflows. It leverages LLM-driven evolutionary optimization to automatically modify and evaluate workflow code (such as `graph.py` and `prompt.py`), aiming to improve performance on a specified benchmark.

AFlow supports multi-round iterative optimization. In each round, it generates new workflow variants based on historical performance, validates them on evaluation sets, and selects the best-performing solution.

---

## Key Features

- **Automated Evolutionary Optimization**: Uses LLM feedback to automatically modify workflow structure and prompts.
- **Multi-round Iteration**: Supports multiple optimization rounds with automatic convergence detection.
- **Multi-benchmark Support**: Works with various benchmarks such as humaneval, math, and more.
- **Highly Extensible**: Supports custom operators, agents, and evaluators.

---

## Typical Usage

A typical optimization workflow (see `run_aflow_optimize.py`):

```python
from mas_arena.evaluators.humaneval_evaluator import HumanEvalEvaluator
from mas_arena.optimizers.aflow_optimizer import AFlowOptimizer
from mas_arena.optimizers.aflow.aflow_experimental_config import EXPERIMENTAL_CONFIG
from mas_arena.agents import AgentSystemRegistry

# 1. Configure agents
optimizer_agent = AgentSystemRegistry.get("single_agent", {"model_name": "gpt-4o-mini", ...})
executor_agent = AgentSystemRegistry.get("single_agent", {"model_name": "gpt-4o-mini", ...})

# 2. Load evaluator
evaluator = HumanEvalEvaluator("humaneval", {})

# 3. Create optimizer
optimizer = AFlowOptimizer(
    graph_path="mas_arena/configs/aflow",
    optimized_path="example/aflow/humaneval/optimization2",
    optimizer_agent=optimizer_agent,
    executor_agent=executor_agent,
    validation_rounds=1,
    eval_rounds=1,
    max_rounds=3,
    **EXPERIMENTAL_CONFIG["humaneval"]
)

optimizer.setup()
optimizer.optimize(evaluator)
optimizer.test(evaluator)
```

---

## Parameter Overview

| Parameter            | Description                                                                 | Type/Default        |
|----------------------|-----------------------------------------------------------------------------|---------------------|
| `graph_path`         | Initial workflow directory, must contain `graph.py` and `prompt.py`          | str                 |
| `optimized_path`     | Directory to save optimization results, defaults to `graph_path`             | str/None            |
| `optimizer_agent`    | LLM agent for optimization (required)                                        | AgentSystem         |
| `executor_agent`     | LLM agent for execution, defaults to optimizer_agent                         | AgentSystem/None    |
| `question_type`      | Problem type (e.g., code, math), can be set via `EXPERIMENTAL_CONFIG`        | str                 |
| `operators`          | List of available operator names, see `aflow_experimental_config.py`         | List[str]           |
| `validation_rounds`  | Number of validation runs per round for stable evaluation                    | int/5               |
| `eval_rounds`        | Number of final test runs                                                    | int/3               |
| `max_rounds`         | Maximum number of optimization rounds                                        | int/20              |
| `check_convergence`  | Whether to stop early if performance plateaus                                | bool/True           |
| `initial_round`      | Starting round (supports resuming)                                           | int/0               |
| ...                  | See source code for more details                                             |                     |

---

## Workflow

1. **Initialization**: Copies the initial workflow to the `round_0` directory and prepares the optimization environment.
2. **Baseline Evaluation**: Evaluates the initial workflow multiple times and records the score.
3. **Evolutionary Optimization**:
   - Selects the best-performing workflow from history as the parent.
   - Generates an LLM prompt using historical experience and operator descriptions.
   - LLM generates new workflow code and modification description.
   - Validates the new workflow and records experience.
   - Checks for convergence or maximum rounds.
4. **Final Testing**: Evaluates the best workflow on the test set multiple times and outputs the final score.

---

## Configuration

### Benchmark Configuration

Available benchmarks are auto-registered in `mas_arena.evaluators`. Common options include:

- humaneval
- math
- mbpp
- mmlu_pro
- swebench
- gsm8k
- hotpotqa
- ifeval
- aime
- bbh
- drop
- gaia

You can check available options with:
```python
from mas_arena.evaluators import BENCHMARKS
print(BENCHMARKS.keys())
```

### Operator Configuration

Recommended operator sets for each benchmark are defined in `mas_arena/optimizers/aflow/aflow_experimental_config.py`, e.g.:

```python
EXPERIMENTAL_CONFIG = {
    "humaneval": {
        "question_type": "code",
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"]
    },
    "math": {
        "question_type": "math",
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    }
}
```

---

## FAQ

- **Path Issues**: Ensure `graph_path` contains both `graph.py` and `prompt.py`.
- **Agent Configuration**: Make sure API_KEY, API_BASE, model_name, etc. are set correctly.
- **Early Stopping**: To disable automatic convergence detection, set `check_convergence=False`.

---

## References

- Main entry script: `run_aflow_optimize.py`
- Optimizer source: `mas_arena/optimizers/aflow_optimizer.py`
- Operator config: `mas_arena/optimizers/aflow/aflow_experimental_config.py`
- Benchmark registration: `mas_arena/evaluators/`

---

For more advanced usage, debugging tips, or extension guides, please refer to the source code or contact the developers.
