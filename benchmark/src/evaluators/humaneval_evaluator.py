"""
HumanEval Evaluator

This module provides a standalone evaluator for HumanEval problems.
"""

import time
from typing import Dict, Any, Tuple
from pathlib import Path
import re
import traceback
from threading import Thread

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from benchmark.src.evaluators.utils.sanitize import sanitize, code_extract

class HumanEvalEvaluator:
    """Evaluator for HumanEval problems"""
    
    def __init__(self, name: str = "humaneval", config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        
        # Set up paths
        self.data_path = config.get("data_path", f"benchmark/data/{name}_test.jsonl")
        self.log_path = config.get("log_path", f"benchmark/data/results/{name.upper()}")
        
        # Create log directory
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize run evaluator
        self.run_evaluator = RunEvaluator()
        
    class TimeoutError(Exception):
        """Timeout error for code execution"""
        pass
        
    def run_with_timeout(self, func, args, timeout=60):
        """Run function with timeout"""
        result = []
        def target():
            result.append(func(*args))
            
        thread = Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise self.TimeoutError()
            
        return result[0]
        
    def check_solution(self, solution: str, test: str, entry_point: str) -> bool:
        """Check if solution passes the test"""
        try:
            # Create test environment
            test_env = {}
            
            # Execute solution
            exec(solution, test_env)
            func = test_env[entry_point]
            
            # Execute test
            exec(test, test_env)
            check_func = test_env["check"]
            
            # Run test with timeout
            return check_func(func)
            
        except Exception as e:  # noqa: F841
            if self.config.get("verbose", False):
                traceback.print_exc()
            return False
            
    # def extract_code(self, text: str) -> str:
    #     """Extract code from text"""
    #     # First try to extract using sanitize
    #     try:
    #         return sanitize(text)
    #     except:  # noqa: E722
    #         # Fallback to code_extract
    #         return code_extract(text)

    def extract_code(self, text: str) -> str:
        """
        Extract python code fenced by ```python ... ```.

        Priority:
        1. Block that appears under a heading containing 'Final Code'
        2. First python block
        3. Fallback to sanitize / code_extract
        """
        # 1) 找到所有 ```python ... ``` 代码块
        blocks = re.findall(r"```python\\s*([\\s\\S]*?)```", text, re.IGNORECASE)

        if blocks:
            # 尝试定位 'Final Code' 区块
            final_block = None
            # split text by headings
            for block in blocks:
                # 通过查看 block 前面的 100 字符是否包含 'Final Code'
                idx = text.find(block)
                prefix = text[max(0, idx-100):idx].lower()
                if "final code" in prefix:
                    final_block = block
                    break
            return (final_block or blocks[0]).strip()

        # 2) fallback – 保持向后兼容
        try:
            return sanitize(text)
        except Exception:  # noqa: E722
            return code_extract(text)
            
    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        """Calculate score for the prediction"""
        # For HumanEval, scoring is done in check_solution
        return 1.0 if self.check_solution(prediction, expected_output, "solve") else 0.0, prediction
        
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: float) -> Run:
        """Create a LangSmith run for evaluation"""
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["test"],
                "score": score,
                "passed": score == 1.0,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
        
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a problem given the agent's response"""
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")
        
        # Extract and clean code
        extracted_answer = self.extract_code(final_answer)
        
        # Calculate score
        score = 1.0 if self.check_solution(extracted_answer, problem["test"], problem["entry_point"]) else 0.0
        
        # Create LangSmith run
        run = self.create_run(problem, final_answer, extracted_answer, score)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)
        
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "run_evaluation": run_evaluation,
        }