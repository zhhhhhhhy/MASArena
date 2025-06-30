# 🚀 Extending MASArena Framework

A comprehensive guide to extending MASArena with custom Multi-Agent Systems and Evaluators.

## 📋 Table of Contents

- [🤖 Multi-Agent System Extension](#-multi-agent-system-extension)
  - [📋 Implementation Requirements](#-implementation-requirements)
  - [📝 Implementation Steps](#-implementation-steps)
  - [⚡ Advanced Features](#-advanced-features)
  - [💡 Complete Example](#-complete-example)
- [🎯 Evaluator Extension](#-evaluator-extension)
  - [🔧 Basic Implementation](#-basic-implementation)
  - [⚡ Advanced Features](#-advanced-features-1)
  - [💻 Code Evaluation](#-code-evaluation)
  - [📖 Complete Examples](#-complete-examples)
- [✅ Best Practices](#-best-practices)
- [🚨 Common Issues](#-common-issues)

---

## 🤖 Multi-Agent System Extension

### 📋 Implementation Requirements

**✅ Essential Requirements:**
- Extend `AgentSystem` base class
- Implement `run_agent()` method (abstract method - required)
- Include `evaluator` in config during initialization
- Return proper message format with usage metadata
- Register with `AgentSystemRegistry`

**💡 Optional but Recommended:**
- Implement `_create_agents()` for tool integration support
- Use `self.format_prompt` for benchmark-specific formatting
- Handle async execution properly if needed

### 📝 Implementation Steps

#### Step 1: Basic Class Structure

```python
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry
from typing import Dict, Any, Optional

class YourMAS(AgentSystem):
    def __init__(self, name: str = "your_mas", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        # ⚠️ IMPORTANT: config must include 'evaluator' key
        self.workers = None
        # Additional initialization here
```

#### Step 2: Implement Core Method

```python
def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    🎯 Main method that processes problems - REQUIRED
    
    Args:
        problem: Problem dictionary with 'problem' key
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with 'messages' and 'final_answer'
    """
    problem_text = problem["problem"]
    
    # Your agent logic here
    response = self.solve_problem(problem_text)
    
    # 📝 Ensure proper message format for metrics collection
    message = {
        'content': response,
        'name': 'your_agent',
        'role': 'assistant',
        'usage_metadata': {}  # Include token usage info
    }
    
    return {
        "messages": [message],
        "final_answer": response,
        # Add any additional data you need
    }
```

#### Step 3: Tool Integration Support (Optional)

```python
def _create_agents(self, problem_input: Optional[Any] = None) -> Dict[str, Any]:
    """
    🔧 Enable tool integration - OPTIONAL
    
    Returns:
        - Multi-agent: {"workers": [agent1, agent2, ...]}
        - Single agent: {"agent_name": agent_instance}
    """
    agents = []
    for i in range(self.config.get("num_agents", 2)):
        agent = AgentNode(
            name=f"solver_{i+1}",
            model_name=self.config.get("model_name", "gpt-4o-mini"),
            prompt=f"You are solver {i+1}. {self.format_prompt}"
        )
        agents.append(agent)
    
    return {"workers": agents}
```

#### Step 4: Registration

```python
# 📋 Register your MAS with optional default configuration
AgentSystemRegistry.register(
    "your_mas", 
    YourMAS, 
    num_agents=3,
    model_name="gpt-4o-mini"
)
```

### ⚡ Advanced Features

#### 🎨 Format Prompt Integration

```python
def _get_system_prompt(self) -> str:
    """Use benchmark-specific formatting requirements"""
    return f"""You are an expert problem solver.

{self.format_prompt}

Solve the problem step by step and provide your final answer clearly."""
```

#### 🔄 Async Execution Handling

```python
def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Handle async operations in sync method"""
    
    async def async_solve():
        result = await self.async_problem_solving(problem)
        return result
    
    # Convert async to sync
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    result = loop.run_until_complete(async_solve())
    return result
```

#### 🤖 AgentNode Pattern

```python
class AgentNode:
    """Standard agent node for tool integration"""
    
    def __init__(self, name: str, model_name: str, prompt: str):
        self.name = name  # Required for tool binding
        self.model_name = model_name
        self.prompt = prompt
        self.llm = ChatOpenAI(model=model_name)  # Required for tools
        
    def solve(self, problem: str):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": problem}
        ]
        response = self.llm.invoke(messages)
        response.name = self.name  # For metrics tracking
        return response
```

### 💡 Complete Example

```python
import os
import asyncio
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

class AgentNode:
    def __init__(self, name: str, model_name: str, prompt: str):
        self.name = name
        self.model_name = model_name
        self.prompt = prompt
        self.llm = ChatOpenAI(model=model_name)

    def solve(self, problem: str):
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": problem}
        ]
        response = self.llm.invoke(messages)
        response.name = self.name
        return response

class MultiSolverMAS(AgentSystem):
    """🧠 Multi-solver MAS with result aggregation"""
    
    def __init__(self, name: str = "multi_solver", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.num_agents = self.config.get("num_agents", 3)

    def _create_agents(self, problem_input: Optional[Any] = None) -> Dict[str, List]:
        """Create multiple specialized solver agents"""
        agents = []
        specializations = ["analytical", "creative", "systematic"]
        
        for i in range(self.num_agents):
            specialty = specializations[i % len(specializations)]
            agent = AgentNode(
                name=f"{specialty}_solver",
                model_name=self.config.get("model_name", "gpt-4o-mini"),
                prompt=f"""You are a {specialty} problem solver.
{self.format_prompt}

Use {specialty} thinking to approach and solve problems."""
            )
            agents.append(agent)
        
        return {"workers": agents}

    def _aggregate_solutions(self, solutions: List[str]) -> str:
        """Aggregate multiple solutions"""
        if len(solutions) == 1:
            return solutions[0]
        
        # Simple implementation - you can add sophisticated logic
        return solutions[0]

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """🚀 Main execution method"""
        problem_text = problem["problem"]
        workers_dict = self._create_agents(problem_text)
        agents = workers_dict["workers"]
        
        solutions = []
        messages = []
        
        # Collect solutions from all agents
        for agent in agents:
            response = agent.solve(problem_text)
            solutions.append(response.content)
            messages.append(response)
        
        # Aggregate results
        final_answer = self._aggregate_solutions(solutions)
        
        return {
            "messages": messages,
            "final_answer": final_answer,
            "individual_solutions": solutions,
            "agent_count": len(agents)
        }

# 📋 Register with configuration
AgentSystemRegistry.register(
    "multi_solver", 
    MultiSolverMAS, 
    num_agents=3,
    model_name="gpt-4o-mini"
)
```

---

## 🎯 Evaluator Extension

### 🔧 Basic Implementation

#### Step 1: Basic Structure

```python
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark

@register_benchmark(
    name="your_benchmark",
    normalization_keys={
        "id": "problem_id",           # 🗝️ Map your data fields
        "problem": "question",        # to standard format
        "solution": "expected_answer"
    }
)
class YourEvaluator(BaseEvaluator):
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        # 🎛️ Custom configuration
        self.case_sensitive = config.get("case_sensitive", False)
        
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 Main evaluation method - REQUIRED"""
        final_answer = self.extract_final_answer(run_result.get("messages", []))
        score = 1 if self.verify_answer(final_answer, problem["solution"]) else 0
        
        return {
            "final_answer": final_answer,
            "extracted_answer": final_answer,
            "score": score,
            "passed": score == 1
        }
    
    def verify_answer(self, prediction: str, reference: str) -> bool:
        """✅ Answer verification logic - OPTIONAL but recommended"""
        if not self.case_sensitive:
            prediction = prediction.lower()
            reference = reference.lower()
        return prediction.strip() == reference.strip()
```

### ⚡ Advanced Features

#### 🔍 Answer Extraction

```python
def extract_final_answer(self, messages: list) -> str:
    """Extract final answer from agent conversation messages"""
    if not messages:
        return ""
    
    last_msg = messages[-1]
    
    # Handle different message formats
    if isinstance(last_msg, tuple) and len(last_msg) > 1:
        content = last_msg[1]  # (agent_name, content)
    elif hasattr(last_msg, "content"):
        content = last_msg.content  # AIMessage
    elif isinstance(last_msg, dict):
        content = last_msg.get("content", "")  # Dict format
    else:
        content = str(last_msg)
    
    # 🔎 Look for answer patterns
    patterns = [
        r"(?:final\s+)?answer:\s*(.+)",
        r"solution:\s*(.+)",
        r"result:\s*(.+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return content.strip()
```

#### 📊 LangSmith Integration

```python
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
import uuid
import time

def __init__(self, name: str, config: Dict[str, Any] = None):
    super().__init__(name, config)
    self.run_evaluator = RunEvaluator()

def create_run(self, problem: Dict[str, Any], final_answer: str, score: float) -> Run:
    """Create LangSmith run for tracking"""
    return Run(
        id=str(uuid.uuid4()),
        name=f"{self.name.upper()}_Evaluation",
        inputs={"problem": problem["problem"]},
        outputs={
            "prediction": final_answer,
            "expected": problem["solution"],
            "score": score,
            "passed": score == 1
        },
        run_type="evaluation",
        start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        trace_id=str(uuid.uuid4())
    )
```

#### 🔢 Mathematical Processing

```python
import re
from typing import Optional

def extract_number(self, text: str) -> Optional[float]:
    """Extract numerical answers from text"""
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
    if matches:
        last_number = matches[-1].replace(",", "")
        try:
            return float(last_number)
        except ValueError:
            return None
    return None

def verify_numerical_answer(self, prediction: str, reference: str, tolerance: float = 1e-6) -> bool:
    """Verify numerical answers with tolerance"""
    pred_num = self.extract_number(prediction)
    ref_num = self.extract_number(reference)
    
    if pred_num is None or ref_num is None:
        return str(prediction).strip() == str(reference).strip()
    
    return abs(pred_num - ref_num) <= tolerance
```

### 💻 Code Evaluation

For code evaluation tasks, extend `BaseCodeEvaluator`:

```python
from mas_arena.evaluators.base_code_evaluator import BaseCodeEvaluator
from threading import Thread
from typing import Tuple, Callable, Any
import re

@register_benchmark(
    name="your_code_benchmark",
    normalization_keys={
        "id": "task_id",
        "problem": "prompt",
        "solution": "canonical_solution",
        "test": "test",
        "entry_point": "entry_point"
    }
)
class YourCodeEvaluator(BaseCodeEvaluator):
    
    class TimeoutError(Exception):
        """⏱️ Custom timeout exception"""
        pass
    
    def extract_code(self, text: str) -> str:
        """🔍 Extract Python code from response"""
        # Look for validated code section first
        validated = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.I)
        if validated:
            return validated.group(1).strip()
        
        # Look for any python code block
        fenced = re.search(r"```python\s*([\s\S]*?)```", text, re.I)
        if fenced:
            return fenced.group(1).strip()
        
        return text.strip()
    
    def run_with_timeout(self, func: Callable, args: tuple, timeout: int = 60) -> Any:
        """Execute function with timeout protection"""
        result = []
        exception = []
        
        def target():
            try:
                result.append(func(*args))
            except BaseException as e:
                exception.append(e)
        
        thread = Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise self.TimeoutError("Execution timed out")
        
        if exception:
            raise exception[0]
        
        return result[0] if result else None
    
    def check_solution(self, code: str, test: str, entry_point: str) -> Tuple[bool, str]:
        """✅ Check if code solution passes tests - REQUIRED for code evaluators"""
        try:
            # 🔒 Create isolated execution environment
            env = {}
            
            # Execute the solution code
            exec(code, env)
            candidate_fn = env[entry_point]
            
            # Execute test code
            exec(test, env)
            check_fn = env["check"]
            
            # Run tests with timeout
            self.run_with_timeout(check_fn, (candidate_fn,), timeout=60)
            return True, "All tests passed"
            
        except self.TimeoutError:
            return False, "⏰ Execution timed out"
        except Exception as e:
            return False, f"❌ Test failed: {str(e)}"
```

### 📖 Complete Examples

#### 📝 Text Evaluator Example

```python
import re
from typing import Dict, Any, Optional
from mas_arena.evaluators.base_evaluator import BaseEvaluator
from mas_arena.evaluators.registry import register_benchmark

@register_benchmark(
    name="smart_text",
    normalization_keys={
        "id": "question_id",
        "problem": "question_text",
        "solution": "correct_answer"
    }
)
class SmartTextEvaluator(BaseEvaluator):
    """🧠 Text evaluator with multiple matching strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.case_sensitive = config.get("case_sensitive", False)
        self.exact_match = config.get("exact_match", False)
        self.fuzzy_threshold = config.get("fuzzy_threshold", 0.8)
    
    def extract_final_answer(self, messages: list) -> str:
        """🔍 Multi-strategy answer extraction"""
        if not messages:
            return ""
        
        last_msg = messages[-1]
        content = self._get_message_content(last_msg)
        
        # Try multiple extraction patterns
        patterns = [
            r"(?:final\s+)?answer:\s*(.+?)(?:\n|$)",
            r"(?:the\s+)?solution\s+is:\s*(.+?)(?:\n|$)",
            r"(?:therefore|thus|so),?\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # Fallback: return last sentence
        sentences = content.split('.')
        return sentences[-1].strip() if sentences else content.strip()
    
    def _get_message_content(self, message) -> str:
        """Extract content from various message formats"""
        if hasattr(message, "content"):
            return message.content
        elif isinstance(message, dict):
            return message.get("content", "")
        elif isinstance(message, tuple):
            return message[1] if len(message) > 1 else ""
        return str(message)
    
    def normalize_text(self, text: str) -> str:
        """🔧 Text normalization for comparison"""
        text = text.strip()
        
        if not self.case_sensitive:
            text = text.lower()
        
        # Remove extra whitespace and trailing punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.!?]+$', '', text)
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """📊 Calculate text similarity"""
        text1, text2 = self.normalize_text(text1), self.normalize_text(text2)
        
        if text1 == text2:
            return 1.0
        
        # Simple character overlap ratio
        chars1, chars2 = set(text1), set(text2)
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def verify_answer(self, prediction: str, reference: str) -> bool:
        """🎯 Multi-strategy answer verification"""
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        # Exact match
        if self.exact_match:
            return pred_norm == ref_norm
        
        # Containment check
        if ref_norm in pred_norm or pred_norm in ref_norm:
            return True
        
        # Fuzzy matching
        similarity = self.calculate_similarity(pred_norm, ref_norm)
        return similarity >= self.fuzzy_threshold
    
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """🚀 Main evaluation with detailed feedback"""
        final_answer = self.extract_final_answer(run_result.get("messages", []))
        extracted_answer = self.normalize_text(final_answer)
        
        is_correct = self.verify_answer(final_answer, problem["solution"])
        score = 1 if is_correct else 0
        similarity = self.calculate_similarity(final_answer, problem["solution"])
        
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "passed": is_correct,
            "similarity": similarity,
            "evaluation_method": "exact" if extracted_answer == self.normalize_text(problem["solution"]) else "fuzzy"
        }
```

---

## ✅ Best Practices

### 🚀 Performance & Security

- **⚡ Batch Processing**: Implement `batch_evaluate()` for better performance
- **⏱️ Timeout Handling**: Always set timeouts for external calls and code execution
- **🔍 Input Validation**: Validate all inputs before processing
- **🛡️ Error Handling**: Implement comprehensive exception handling
- **📝 Logging**: Add detailed logging for debugging and monitoring

### 🧪 Testing & Validation

- **🎯 Unit Tests**: Test individual components thoroughly
- **🔄 Integration Tests**: Test full evaluation pipeline
- **⚠️ Edge Cases**: Test with malformed inputs and edge cases
- **📊 Performance Tests**: Benchmark evaluation speed for large datasets

---

## 🚨 Common Issues

### 📋 Implementation Checklist

**For MAS Extensions:**
- [ ] ✅ Config includes `evaluator` key
- [ ] 📊 Messages have `usage_metadata` for token tracking
- [ ] 🏷️ Agents have `name` and `llm` attributes (for tool integration)
- [ ] ⚡ `run_agent` method is synchronous
- [ ] 📤 Return format includes `messages` and `final_answer`
- [ ] 📋 Proper registration with `AgentSystemRegistry`

**For Evaluator Extensions:**
- [ ] 🎯 Used `@register_benchmark` decorator
- [ ] ✅ Implemented `evaluate` method
- [ ] 🗝️ Proper normalization_keys mapping
- [ ] 🛡️ Error handling for malformed inputs
- [ ] ⏱️ Timeout handling for long operations

### ⚠️ Common Mistakes

| Issue | Solution |
|-------|----------|
| Making `run_agent` async | Keep it synchronous, wrap async calls internally |
| Missing evaluator in config | Always include `evaluator` key in MAS config |
| No usage_metadata in messages | Include token usage info in message objects |
| Hardcoded model names | Use config parameters for flexibility |
| No timeout for code execution | Always set execution timeouts |
| Ignoring message format variations | Handle multiple message formats (tuple, dict, object) |

### 🔧 Debugging Tips

1. **📝 Check Logs**: Review evaluator logs in the configured log directory
2. **🔍 Validate Config**: Ensure all required configuration keys are present
3. **💬 Test Message Format**: Verify message objects have expected attributes
4. **🎯 Run Simple Cases**: Start with simple test cases before complex ones
5. **📊 Monitor Performance**: Check execution times and memory usage

---

💡 **Pro Tip**: This guide provides the essential information for extending MASArena. Start with simple implementations and gradually add advanced features as needed.
