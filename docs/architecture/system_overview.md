# Multi-Agent Benchmark System Architecture

This document provides a detailed overview of the Multi-Agent Benchmark system's architecture. It explains the core components, their interactions, and the overall data flow when running a benchmark.

## High-Level Architecture

The system is designed to be modular and extensible, allowing for easy addition of new agent systems and benchmarks. The core components are the `BenchmarkRunner`, `AgentSystem`, and `Evaluator`. The `BenchmarkRunner` orchestrates the process, while the `AgentSystem` encapsulates the logic for both solving a problem and evaluating its own solution.

```mermaid
graph TD
    subgraph User Interaction
        A[run_benchmark.sh]
    end

    subgraph Core Orchestration
        B[main.py]
        C[BenchmarkRunner]
    end

    subgraph Agent System Abstraction
        D[agents.create_agent_system]
        E[agents.AgentSystemRegistry]
        F[agents.base.AgentSystem]
        G[agents.run_agent]
    end
    
    subgraph Concrete Agent Systems
        direction LR
        H[MetaGPT]
        I[AgentVerse]
        J[Swarm]
        K[...]
    end

    subgraph Evaluator Abstraction
        L[evaluators.base_evaluator.BaseEvaluator]
        M[evaluators.evaluate]
    end

    subgraph Concrete Evaluators
        direction LR
        N[HumanEvalEvaluator]
        O[MBPPEvaluator]
        P[SWEBenchEvaluator]
        Q[...]
    end
    
    subgraph Data
        R[Benchmark Datasets]
    end
    
    subgraph Results
        S[Results]
        T[Metrics]
    end

    A -- "Executes with args (agent, benchmark)" --> B
    B -- "Instantiates & calls" --> C
    
    C -- "Calls with agent name" --> D
    D -- "Looks up in" --> E
    E -- "Instantiates" --> F
    
    F -- "Is subclassed by" --> H
    F -- "Is subclassed by" --> I
    F -- "Is subclassed by" --> J
    F -- "Is subclassed by" --> K

    F -- "Initializes" --> L
    L -- "Is subclassed by" --> N
    L -- "Is subclassed by" --> O
    L -- "Is subclassed by" --> P
    L -- "Is subclassed by" --> Q
    
    C -- "Loads" --> R
    C -- "For each problem in dataset, calls" --> F
    
    F -- "evaluate(problem) calls" --> G
    G -- "Gets result, then calls" --> M
    
    C -- "Saves" --> S
    C -- "Saves" --> T

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#ccf,stroke:#333,stroke-width:2px
```

## Execution Workflow

The following sequence diagram illustrates the step-by-step workflow when a benchmark is executed. A key design choice is that the `AgentSystem` is responsible for its own evaluation. It creates an appropriate `Evaluator` during its initialization and uses it to score the solutions it generates.

```mermaid
sequenceDiagram
    participant User
    participant run_benchmark.sh
    participant main.py
    participant BenchmarkRunner
    participant AgentSystem
    participant Evaluator

    User->>run_benchmark.sh: Execute with args
    run_benchmark.sh->>main.py: Run Python script
    main.py->>BenchmarkRunner: Instantiate(results_dir, metrics_dir)
    main.py->>BenchmarkRunner: run(benchmark, agent_system, ...)

    BenchmarkRunner->>AgentSystem: create_agent_system(agent_system, config)
    activate AgentSystem
    
    AgentSystem->>Evaluator: Instantiate(evaluator_name)
    activate Evaluator
    Note over AgentSystem, Evaluator: Agent creates its own Evaluator
    deactivate Evaluator
    
    BenchmarkRunner-->>AgentSystem: Returns created agent instance

    loop For each problem in dataset
        BenchmarkRunner->>AgentSystem: evaluate(problem)
        AgentSystem->>AgentSystem: run_agent(problem)
        Note right of AgentSystem: Core agent logic to generate solution
        AgentSystem->>Evaluator: evaluate(solution, ground_truth)
        activate Evaluator
        Evaluator-->>AgentSystem: Return score & metrics
        deactivate Evaluator
        AgentSystem-->>BenchmarkRunner: Return evaluation results
    end
    
    deactivate AgentSystem
    
    BenchmarkRunner->>main.py: Return summary
    main.py->>User: Print summary and exit
```

## Core Components Decomposition

The framework's modularity comes from its use of abstract base classes and registries for dynamic discovery.

### Agent Systems

All agent systems inherit from the `AgentSystem` abstract base class. This ensures they conform to a common interface, which includes the `run_agent()` and `evaluate()` methods. The `AgentSystemRegistry` is used to discover and list available agents.

```mermaid
classDiagram
    direction LR
    class AgentSystem {
        <<Abstract>>
        +name: str
        +config: dict
        +evaluator: BaseEvaluator
        +run_agent(problem) dict
        +evaluate(problem) dict
    }

    class MetaGPT {
    }
    
    class AgentVerse {
    }
    
    class Swarm {
    }

    AgentSystem <|-- MetaGPT
    AgentSystem <|-- AgentVerse
    AgentSystem <|-- Swarm
```

### Evaluators

Similarly, all evaluators inherit from a `BaseEvaluator` class (though not strictly enforced as an ABC in the current implementation, it serves this role conceptually). The `AVAILABLE_EVALUATORS` dictionary in `benchmark/src/evaluators/__init__.py` acts as a registry.

```mermaid
classDiagram
    direction LR
    class BaseEvaluator {
        <<Interface>>
        +name: str
        +config: dict
        +evaluate(prediction, expected) dict
    }

    class HumanEvalEvaluator {
    }
    
    class MBPPEvaluator {
    }
    
    class SWEBenchEvaluator {
    }

    BaseEvaluator <|-- HumanEvalEvaluator
    BaseEvaluator <|-- MBPPEvaluator
    BaseEvaluator <|-- SWEBenchEvaluator
```

## Extensibility

Adding a new agent or evaluator to the system is straightforward.

### Adding a New Agent
1.  Create a new Python file in `benchmark/src/agents/`.
2.  Implement a new class that inherits from `agents.base.AgentSystem`.
3.  Implement the abstract `run_agent()` method with the agent's unique logic.
4.  Register the new agent in `benchmark/src/agents/__init__.py` by adding it to the `AVAILABLE_AGENT_SYSTEMS` dictionary and the `__all__` list.

### Adding a New Evaluator
1.  Create a new Python file in `benchmark/src/evaluators/`.
2.  Implement a new class that provides an `evaluate()` method.
3.  Register the new evaluator in `benchmark/src/evaluators/__init__.py` by adding it to the `AVAILABLE_EVALUATORS` dictionary.