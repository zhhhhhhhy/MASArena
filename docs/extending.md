# Extending a Multi-Agent System

## Implementation Logic

1. Extend `AgentSystem` base class
   - Initialize with name and config
   - Store workers dictionary

2. Implement `_create_agents` method
   - Create specialized `AgentNode` instances
   - Set agent names, models, and prompts
   - Return dictionary of agents

3. Implement `run_agent` method
   - Get agents from `_create_agents`
   - Implement agent interaction logic
   - Return messages and final answer

4. Register system with `AgentSystemRegistry`

### Example MAS Implementation

```python
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

class SimpleMAS(AgentSystem):
    def __init__(self, name: str = "simple_mas", config: Dict[str, Any] = None):
        super().__init__(name, config if config else {})
        self.workers = None

    def _create_agents(self, problem_input: Optional[Any] = None) -> Dict[str, AgentNode]:
        agent = AgentNode(
            name="solver",
            model_name=self.config.get("model_name", "gpt-4"),
            prompt=f"Solve the problem: {self.format_prompt}"
        )
        return {"solver": agent}

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        workers = self._create_agents(problem["problem"])
        response = workers["solver"](problem["problem"])
        return {
            "messages": [response],
            "final_answer": response.content
        }


AgentSystemRegistry.register("simple_mas", SimpleMAS)
```

## Extending an Evaluator

1. Extend `BaseEvaluator` class
   - Initialize with name and config
   - Set up data and log paths
   - Configure logging

2. Implement `evaluate` method
   - Process problem and run result
   - Extract final answer from messages
   - Calculate score
   - Return evaluation results

3. Implement `verify_answer` method (optional)
   - Compare prediction with reference
   - Return boolean result

4. Register evaluator with `register_benchmark` decorator
   - Specify evaluator name
   - Define normalization keys

### Example Evaluator Implementation

```python
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark
<<<<<<< HEAD:docs/extending/extending.md

=======
>>>>>>> 4d2574344847618cc143b8b56c59e7d9587af426:docs/extending.md

@register_benchmark(
   name="simple",
   normalization_keys={
      "id": "id",
      "problem": "problem",
      "solution": "solution"
   }
)
class SimpleEvaluator(BaseEvaluator):
   def __init__(self, name: str, config: Dict[str, Any] = None):
      super().__init__(name, config)

   def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
      final_answer = run_result.get("final_answer", "")
      score = 1 if final_answer == problem["solution"] else 0
      return {
         "final_answer": final_answer,
         "score": score
      }

   def verify_answer(self, prediction: str, reference: str) -> bool:
      return prediction == reference 
```