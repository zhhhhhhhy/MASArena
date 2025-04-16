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

The computational resource usage for LLM inference is dominated by memory requirements. The framework estimates memory cost using a mathematical model based on model size and usage patterns.

### Memory Cost Formula

The total memory cost for a model $M$ during inference is calculated as:

$$M_{total} = M_{params} + M_{activated} + M_{kv\_cache}$$

Where:
- $M_{total}$ is the total memory usage in bytes
- $M_{params}$ is the parameter memory (full model weights)
- $M_{activated}$ is the activated parameter memory during inference
- $M_{kv\_cache}$ is the key-value cache memory

#### Parameter Memory

$$M_{params} = P \times B_{param}$$

Where:
- $P$ is the number of parameters in the model (e.g., 7 billion, 70 billion)
- $B_{param}$ is the bytes per parameter (typically 2 bytes for FP16 precision)

#### Activated Memory

TODO: The activation memory size should be included in the config dictionary.

$$M_{activated} = M_{params} \times \alpha_{act}$$

TODO: $$M_{activated} = P_{act} \times B_{param}$$
Where:
- $\alpha_{act}$ is the activation ratio, representing the fraction of model 

#### KV Cache Memory

$$M_{kv\_cache} = (T_{in} + T_{out}) \times B_{kv} \times \frac{P}{P_{base}}$$

TODO: revise the base parameter count for scaling.

Where:
- $T_{in}$ is the input token count
- $T_{out}$ is the output token count
- $B_{kv}$ is the base bytes per token in KV cache (typically 8 bytes)
- $P$ is the number of parameters in billions
- $P_{base}$ is a base parameter count for scaling (typically 6 billion)

### Assumptions and Scaling

1. **Model Size & Activation Ratio**: For open-source models, parameter counts are taken from a predefined dictionary. The activation ratio is assumed to be 30% of the parameters. (TODO: The activation memory size should be included in the config dictionary.)
2. **Memory Precision**: All calculations assume FP16 precision (2 bytes per parameter) for inference. More efficient quantization methods (INT8, INT4) would reduce memory requirements.

3. **KV Cache Scaling**: KV cache cost scales linearly with context length (input + output tokens) and proportionally to model size relative to a base size of 6B parameters.

4. **Multi-agent Memory**: For multi-agent systems, total memory includes the sum of all component models' memory requirements, which represents a worst-case scenario where all models are loaded simultaneously.

## Latency and Throughput Estimation

The framework also provides mathematical models for estimating inference latency and throughput, which are critical for evaluating the performance of language models in real-world applications.

### Time to First Token (TTFT)

TTFT represents the time between submitting a prompt and receiving the first token of the response. It is a key metric for user-perceived latency. The TTFT is calculated as:

$$T_{TTFT} = T_{compute} + T_{memory}$$

Where:
- $T_{compute}$ is the computation time required for processing the input prompt
- $T_{memory}$ is the memory access time required for activating the model

These components are calculated as:

$$T_{compute} = \frac{2 \times P \times T_{in}}{F_{effective}}$$

$$T_{memory} = \frac{M_{activated}}{B_{memory}}$$

Where:
- $P$ is the number of parameters in the model
- $T_{in}$ is the input token count
- $F_{effective}$ is the effective FLOPS (floating-point operations per second) of the hardware, calculated as $F_{GPU} \times \eta_{hw}$
- $F_{GPU}$ is the theoretical peak FLOPS of the GPU
- $\eta_{hw}$ is the hardware efficiency factor (typically 0.4 or 40%)
- $M_{activated}$ is the activated memory in bytes (as calculated in the memory estimation section)
- $B_{memory}$ is the memory bandwidth in bytes per second

### Throughput Estimation

Throughput is measured in tokens per second during generation. For decoder-only models, throughput is constrained by both computational limits and memory bandwidth:

$$TP = \min(TP_{compute}, TP_{memory})$$

The compute-bound throughput is calculated as:

$$TP_{compute} = \frac{F_{effective}}{O_{token}}$$

Where:
- $F_{effective}$ is the effective FLOPS as defined above
- $O_{token}$ is the operations required per token, approximately $2 \times P$ (2 FLOPs per parameter per token)

The memory-bound throughput is calculated as:

$$TP_{memory} = \frac{B_{memory}}{M_{kv\_per\_token}}$$

Where:
- $B_{memory}$ is the memory bandwidth in bytes per second
- $M_{kv\_per\_token}$ is the KV cache memory required per token, calculated as $\frac{M_{kv\_cache}}{T_{in} + T_{out}}$

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