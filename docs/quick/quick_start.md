# Quick Start

## 1. Setup

We recommend using [uv](https://docs.astral.sh/uv/) for dependency and virtual environment management.

```bash
# Install dependencies
uv sync
```

### Option 1: Using `main.py`

```bash
# Run a math benchmark with a single agent
python main.py --benchmark math --agent-system single_agent --limit 5

# Run with supervisor-based multi-agent system
python main.py --benchmark math --agent-system supervisor_mas --limit 10

# Run with swarm-based multi-agent system
python main.py --benchmark math --agent-system swarm --limit 5
```

### Option 2: Using the Shell Runner

```bash
./run_benchmark.sh math supervisor_mas 10
```

Example output:

```bash
================================================================================
Benchmark Summary
================================================================================
Agent system: swarm
Accuracy: 70.00% (7/10)
Total duration: 335125ms
Results saved to: results/math_swarm_20250616_203434.json
Summary saved to: results/math_swarm_20250616_203434_summary.json

Run visualization:
$ python benchmark/src/visualization/visualize_benchmark.py visualize \
  --summary results/math_swarm_20250616_203434_summary.json
```

---

## 4. Visualizing Agent Interactions

```bash
python benchmark/src/visualization/visualize_benchmark.py visualize \
  --summary results/math_swarm_20250616_203434_summary.json
```

This generates an interactive visualization of agent interactions and benchmark results.

![visualization](../../assets/visual_1.png)

![visualization](../../assets/visual_2.png)


## 📊 Supported Benchmarks

| Benchmark   | Description                  | Dataset File               |
| ----------- | ---------------------------- | -------------------------- |
| `math`      | Mathematical problem solving | `math_test.jsonl`          |
| `humaneval` | Python code generation       | `humaneval_test.jsonl`     |
| `mbpp`      | Python programming problems  | `mbpp_test.jsonl`          |
| `drop`      | Reading comprehension        | `drop_test.jsonl`          |
| `bbh`       | Complex reasoning tasks      | `bbh_test.jsonl`           |
| `ifeval`    | Instruction following        | `ifeval_test.jsonl`        |
| `aime`      | Math competition problems    | `aime_*_test.jsonl`        |
| `mmlu_pro`  | Multi-domain knowledge       | `mmlu_pro_test.jsonl`      |

---

## 🤖 Supported Agent Systems

| Agent System     | File                | Description                         |
| ---------------- | ------------------- | ----------------------------------- |
| `single_agent`   | `single_agent.py`   | Single LLM agent                    |
| `supervisor_mas` | `supervisor_mas.py` | Supervisor-based multi-agent system |
| `swarm`          | `swarm.py`          | Swarm-based agent system            |
| `agentverse`     | `agentverse.py`     | Dynamic recruitment agent system    |
| `chateval`       | `chateval.py`       | Debate-based multi-agent system     |
| `evoagent`       | `evoagent.py`       | Evolutionary agent system           |
| `jarvis`         | `jarvis.py`         | Task-planning agent system          |
| `metagpt`        | `metagpt.py`        | Code generation agent system        |
