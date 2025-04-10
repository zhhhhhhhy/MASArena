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
