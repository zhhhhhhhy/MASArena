# Multi-Agent System Benchmark

A comprehensive framework for benchmarking single and multi-agent systems across various tasks, evaluating their performance, accuracy, and efficiency.

## Hyperparameters

The evaluation framework is built upon four core hyperparameters that encapsulate the trade-offs between performance and cost:

- **$V_A $(Value per Accuracy Unit)**: Represents the economic value gained per unit increase in solution accuracy.
  - Units: $\text{\$/accuracy unit}$

- **$C_R $(Cost per Resource Unit)**: Captures the cost associated with computational resources consumed during task execution.
  - Units: $\text{\$/compute unit}$

- **$C_T $(Cost per Time Unit)**: Reflects the opportunity cost of processing time, accounting for delays in task completion.
  - Units: $\text{\$/time unit}$

- **$V_T $(Value per Throughput Unit)**: Quantifies the value derived from the system's processing capacity, measured as tasks completed per unit time.
  - Units: $\text{\$/task/time unit}$

These hyperparameters allow the evaluation system to translate performance and cost dimensions into a unified economic value.

## Problem Formulation

### System Model

Consider a problem set $T = \{t_1, t_2, ..., t_n\}$, where each task $t_i $has associated accuracy, computational resource usage, and processing time. The systems under evaluation are defined as follows:

- **Single-Agent System (SA)**:
  - A single computational entity processes all tasks sequentially or in batches.
  - Resources are allocated to one decision-making unit.

- **Multi-Agent System (MAS)**:
  - Multiple computational entities operate in parallel to process tasks.
  - Resources are distributed across agents without explicit modeling of communication costs.

#### Performance Variables

For any system $S $(either SA or MAS), we define:

1. **Average Accuracy ($A(S)$)**: The mean accuracy of solutions across all tasks.
   $$
   A(S) = \frac{1}{n} \sum_{i=1}^{n} a(t_i)
   $$
   where $a(t_i) $is the accuracy for task $t_i$.

2. **Total Computational Cost ($R(S)$)**: The total computational resources consumed by the system.
   - For SA:
     $$
     R(SA) = \sum_{i=1}^{n} r(t_i)
     $$
   - For MAS:
     $$
     R(MAS) = \sum_{j=1}^{M} \sum_{i=1}^{n_j} r(t_{ij})
     $$
     where $r(t_{ij}) $is the resource usage for task $t_i $by agent $j$.

3. **Total Latency ($T(S)$)**: The total processing time required to complete all tasks.
   - For SA:
     $$
     T(SA) = \sum_{i=1}^{n} t(t_i)
     $$
   - For MAS:
     $$
     T(MAS) = \max(\text{completion times of all agents})
     $$

4. **Throughput ($P(S)$)**: The number of tasks completed per unit time.
   $$
   P(S) = n / T(S)
   $$

#### Unified Utility Function

The unified utility function translates performance and cost into a single economic value:
$$
U(S) = V_A A(S) + V_T P(S) - C_R R(S) - C_T T(S)
$$
where:
- $V_A A(S)$: Value generated from solution accuracy.
- $V_T P(S)$: Value derived from system throughput.
- $C_R R(S)$: Cost incurred due to computational resource usage.
- $C_T T(S)$: Cost associated with processing time.

This utility function provides a comprehensive measure of system performance while accounting for operational costs.

#### Efficiency Ratio

To normalize performance across different problem scales, we define an efficiency ratio:
$$
E(S) = \frac{V_A A(S) + V_T P(S)}{C_R R(S) + C_T T(S)}
$$
This ratio measures the value generated per unit cost. Systems with $E(S) > 1 $are considered economically efficient.

#### Scalability Coefficient

To evaluate how well a system scales with increasing problem complexity or size, we define the scalability coefficient:
$$
SC(S) = \frac{\Delta U(S)}{\Delta n}
$$
where $n $represents the number of tasks or problem size. This coefficient quantifies the change in utility as the problem size increases.

### Implementation Considerations

#### Open-Source Models
For open-source models, metrics are calculated based on:
1. **Latency ($T$)**:
   - Measured in seconds per task
   - Includes model inference time and any preprocessing/postprocessing
   - Hardware configuration (CPU/GPU type, memory) must be specified

2. **Computational Cost ($R$)**:
   - Based on input tokens ($t_{in}$) and output tokens ($t_{out}$)
   - Model configuration (parameters, architecture) affects cost
   - Hardware utilization (FLOPs, memory usage) is tracked

3. **Throughput ($P$)**:
   - Tasks completed per second
   - Measured under consistent hardware conditions
   - Batch processing capabilities are considered

#### Closed-Source Models (API Services)
For closed-source models accessed via API:
1. **Cost ($C$)**:
   - Including both latency and computational cost
   - Direct API costs in dollars

2. **Throughput**:
   - From the leaderboard data or real experiments
  
### Scenarios

The framework supports various scenarios by adjusting the four core hyperparameters to reflect different operational contexts:

1. **High-Accuracy Critical Scenario**
   - $V_A$: High (e.g., medical diagnosis, financial forecasting)
   - $C_R$: Medium (willing to invest in computational resources)
   - $C_T$: Low (accuracy prioritized over speed)
   - $V_T$: Low (throughput less important than precision)

2. **Time-Sensitive Scenario**
   - $V_A$: Medium (acceptable trade-off with speed)
   - $C_R$: High (willing to use more resources for speed)
   - $C_T$: High (time is critical)
   - $V_T$: High (high throughput is essential)

3. **Resource-Constrained Scenario**
   - $V_A$: Medium (balanced accuracy requirements)
   - $C_R$: High (resource usage is expensive)
   - $C_T$: Medium (moderate time sensitivity)
   - $V_T$: Medium (moderate throughput needs)

4. **Balanced Performance Scenario**
   - $V_A$: Medium (standard accuracy requirements)
   - $C_R$: Medium (standard resource costs)
   - $C_T$: Medium (standard time sensitivity)
   - $V_T$: Medium (standard throughput needs)

These scenarios can be used to evaluate system performance under different operational conditions and help identify the optimal system configuration for specific use cases.


-----

# Metrics Estimation


## Metrics Evaluation Output

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

## Memory Usage Estimation
`benchmark/src/instrumentation/memory_instrumentation.py`
The computational resource usage for LLM inference is dominated by memory requirements. The framework estimates memory cost using a mathematical model based on model size and usage patterns.

### Memory Cost Formula

The total memory cost for a model $M$ during inference is calculated as:

$$
M_{\mathrm{total}} = M_{\mathrm{params}} + M_{\mathrm{activated}} + M_{\mathrm{kv\_cache}}
$$

Where:
- $M_{\mathrm{total}}$ is the total memory usage in bytes
- $M_{\mathrm{params}}$ is the parameter memory (full model weights)
- $M_{\mathrm{activated}}$ is the activated parameter memory during inference
- $M_{\mathrm{kv\_cache}}$ is the key-value cache memory

#### Parameter Memory

The memory allocated for all model parameters is:

$$
M_{\mathrm{params}} = P \times 10^9 \times B_{\mathrm{param}}
$$

Where:
- $P$ is the total number of parameters, in billions
- $B_{\mathrm{param}}$ is the number of bytes per parameter (e.g., 2 for FP16)

#### Activated Memory

During inference only a subset of parameters is activated. We denote this by $P_{act}$ (in billions):

$$
M_{\mathrm{activated}} = P_{\mathrm{act}} \times 10^9 \times B_{\mathrm{param}}
$$

Where $P_{\mathrm{act}}$ is the activated parameter count, in billions (provided in configuration).

#### KV Cache Memory 
<!-- todo: review this -->

Key-value cache memory scales with context length and model size. We assume a base token cost $B_{kv}$ = 8 bytes at a reference parameter count $P_{base} = 6$ (billion parameters):

$$
M_{\mathrm{kv\_cache}} = (T_{\mathrm{in}} + T_{\mathrm{out}}) \times B_{\mathrm{kv}} \times \frac{P}{P_{\mathrm{base}}}
$$

Where:
- $T_{\mathrm{in}}$ is the number of input tokens
- $T_{\mathrm{out}}$ is the number of output tokens
- $B_{\mathrm{kv}}$ is base bytes per token in KV cache (8 bytes)
- $P_{\mathrm{base}}$ is reference parameter count (6 billion)
- $P$ is total parameter count in billions

#### Total Memory Usage

Summing these components yields the total memory footprint:

$$
M_{\mathrm{total}} = M_{\mathrm{params}} + M_{\mathrm{activated}} + M_{\mathrm{kv\_cache}}
$$

#### Assumptions
- Parameters are stored in FP16 precision (2 bytes per parameter).
- Activated parameter counts ($P_{\mathrm{act}}$) are defined in `model_data.py`.
- KV cache uses a base rate of 8 bytes per token at 6B parameters and scales linearly with model size.

## Latency and Throughput Estimation

The framework also provides mathematical models for estimating inference latency and throughput, which are critical for evaluating the performance of language models in real-world applications.

### Time to First Token (TTFT)

TTFT represents the latency between prompt submission and receiving the first output token:

$$
T_{\mathrm{TTFT}} = T_{\mathrm{compute}} + T_{\mathrm{memory}}
$$

where

$$
T_{\mathrm{compute}} = \frac{2 \times P \times T_{\mathrm{in}}}{F_{\mathrm{effective}}},
\quad
T_{\mathrm{memory}} = \frac{M_{\mathrm{activated}}}{B_{\mathrm{memory}}}
$$

with:
- $P$: total parameters (billions × $10^9$ parameters)
- $T_{\mathrm{in}}$: number of input tokens
- $F_{\mathrm{effective}} = F_{\mathrm{GPU}} \times \eta_{\mathrm{hw}}$ (effective FLOPS)
- $M_{\mathrm{activated}}$: activated memory in bytes
- $B_{\mathrm{memory}}$: memory bandwidth (bytes/sec)

### Throughput Estimation

Throughput (tokens/sec) is the minimum of compute-bound and memory-bound rates:

$$
\mathrm{TP} = \min(\mathrm{TP}_{\mathrm{compute}}, \mathrm{TP}_{\mathrm{memory}})
$$

where

$$
\mathrm{TP}_{\mathrm{compute}} = \frac{F_{\mathrm{effective}}}{O_{\mathrm{token}}},
\quad
O_{\mathrm{token}} = 2 \times P
$$

$$
\mathrm{TP}_{\mathrm{memory}} = \frac{B_{\mathrm{memory}}}{M_{\mathrm{kv\_per\_token}}},
\quad
M_{\mathrm{kv\_per\_token}} = \frac{M_{\mathrm{kv\_cache}}}{T_{\mathrm{in}} + T_{\mathrm{out}}}
$$

with variables as defined above.

### System-Level Metrics

For multi-agent systems, the framework calculates:

1. **Average Latency**: The weighted average TTFT across all agent interactions.

$$T_{avg} = \frac{\sum_{i=1}^{n} T_{TTFT,i}}{n}$$

Where $n$ is the total number of agent interactions.

2. **Effective Throughput**: The aggregate throughput of the system measured in tasks per second.

$$TP_{system} = \frac{1000}{D_{avg}}$$

Where $D_{avg}$ is the average duration per task in milliseconds.

### Assumptions and Hardware Considerations

1. **Hardware Configuration**: Default values are used if specific hardware details are not provided:
   - Default GPU FLOPS: 20 TFLOPS (typical for mid-range GPUs)
   - Default memory bandwidth: 600 GB/s
   - Default hardware efficiency: 40% of theoretical peak

2. **Computational Model**: The framework assumes that inference has two distinct phases:
   - Prefill phase: Processing the input prompt (computed in TTFT)
   - Generation phase: Producing each output token sequentially

3. **Bottleneck Determination**: Generation throughput is determined by the minimum of compute-bound and memory-bound estimates, as the slower factor becomes the bottleneck.

4. **Task-Level Metrics**: For complete tasks, total inference time is estimated as:
   
$$T_{total} = T_{TTFT} + \frac{T_{out}}{TP}$$

Where $T_{out}$ is the output token count and $TP$ is the throughput in tokens per second.

By applying these mathematical models, the framework provides standardized estimates of performance metrics that enable objective comparisons between different model architectures and agent system designs.

## Custom Metrics

...