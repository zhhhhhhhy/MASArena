# Multi-Agent System Benchmark

A comprehensive framework for benchmarking single and multi-agent systems across various tasks, evaluating their performance, accuracy, and efficiency.

## Hyperparameters

The evaluation framework is built upon four core hyperparameters that encapsulate the trade-offs between performance and cost:

- **$$ V_A $$ (Value per Accuracy Unit)**: Represents the economic value gained per unit increase in solution accuracy.
  - Units: $$ \text{\$/accuracy unit} $$

- **$$ C_R $$ (Cost per Resource Unit)**: Captures the cost associated with computational resources consumed during task execution.
  - Units: $$ \text{\$/compute unit} $$

- **$$ C_T $$ (Cost per Time Unit)**: Reflects the opportunity cost of processing time, accounting for delays in task completion.
  - Units: $$ \text{\$/time unit} $$

- **$$ V_T $$ (Value per Throughput Unit)**: Quantifies the value derived from the system's processing capacity, measured as tasks completed per unit time.
  - Units: $$ \text{\$/task/time unit} $$

These hyperparameters allow the evaluation system to translate performance and cost dimensions into a unified economic value.

## Problem Formulation

### System Model

Consider a problem set $$ T = \{t_1, t_2, ..., t_n\} $$, where each task $$ t_i $$ has associated accuracy, computational resource usage, and processing time. The systems under evaluation are defined as follows:

- **Single-Agent System (SA)**:
  - A single computational entity processes all tasks sequentially or in batches.
  - Resources are allocated to one decision-making unit.

- **Multi-Agent System (MAS)**:
  - Multiple computational entities operate in parallel to process tasks.
  - Resources are distributed across agents without explicit modeling of communication costs.

#### Performance Variables

For any system $$ S $$ (either SA or MAS), we define:

1. **Average Accuracy ($$ A(S) $$)**: The mean accuracy of solutions across all tasks.
   $$
   A(S) = \frac{1}{n} \sum_{i=1}^{n} a(t_i)
   $$
   where $$ a(t_i) $$ is the accuracy for task $$ t_i $$.

2. **Total Computational Cost ($$ R(S) $$)**: The total computational resources consumed by the system.
   - For SA:
     $$
     R(SA) = \sum_{i=1}^{n} r(t_i)
     $$
   - For MAS:
     $$
     R(MAS) = \sum_{j=1}^{M} \sum_{i=1}^{n_j} r(t_{ij})
     $$
     where $$ r(t_{ij}) $$ is the resource usage for task $$ t_i $$ by agent $$ j $$.

3. **Total Latency ($$ T(S) $$)**: The total processing time required to complete all tasks.
   - For SA:
     $$
     T(SA) = \sum_{i=1}^{n} t(t_i)
     $$
   - For MAS:
     $$
     T(MAS) = \max(\text{completion times of all agents})
     $$

4. **Throughput ($$ P(S) $$)**: The number of tasks completed per unit time.
   $$
   P(S) = n / T(S)
   $$

#### Unified Utility Function

The unified utility function translates performance and cost into a single economic value:
$$
U(S) = V_A A(S) + V_T P(S) - C_R R(S) - C_T T(S)
$$
where:
- $$ V_A A(S) $$: Value generated from solution accuracy.
- $$ V_T P(S) $$: Value derived from system throughput.
- $$ C_R R(S) $$: Cost incurred due to computational resource usage.
- $$ C_T T(S) $$: Cost associated with processing time.

This utility function provides a comprehensive measure of system performance while accounting for operational costs.

#### Efficiency Ratio

To normalize performance across different problem scales, we define an efficiency ratio:
$$
E(S) = \frac{V_A A(S) + V_T P(S)}{C_R R(S) + C_T T(S)}
$$
This ratio measures the value generated per unit cost. Systems with $$ E(S) > 1 $$ are considered economically efficient.

#### Scalability Coefficient

To evaluate how well a system scales with increasing problem complexity or size, we define the scalability coefficient:
$$
SC(S) = \frac{\Delta U(S)}{\Delta n}
$$
where $$ n $$ represents the number of tasks or problem size. This coefficient quantifies the change in utility as the problem size increases.

### Implementation Considerations

1. **Hyperparameter Calibration**: The values of $$ V_A, C_R, C_T, V_T $$ must be calibrated based on domain-specific priorities. For example, time-critical applications may assign higher values to $$ C_T $$, while accuracy-critical domains prioritize $$ V_A $$.

2. **Task Normalization**: Ensure that tasks are of equivalent difficulty across systems or apply normalization to account for variations.

3. **Resource Allocation Protocols**: Define clear rules for resource distribution in MAS to ensure fair comparisons with SA systems.

4. **Pareto Frontier Analysis**: Use Pareto efficiency analysis to identify optimal configurations by plotting accuracy versus cost trade-offs.
