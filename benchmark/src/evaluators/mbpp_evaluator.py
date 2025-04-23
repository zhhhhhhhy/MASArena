"""
MBPP Evaluator

This module provides a standalone evaluator for MBPP (Mostly Basic Python Problems) problems.
"""

import time
import threading
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from benchmark.src.evaluators.utils.sanitize import sanitize

class MBPPEvaluator:
    """Evaluator for MBPP problems"""
    
    def __init__(self, name: str = "mbpp", config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        
        # Set up paths
        self.data_path = config.get("data_path", f"benchmark/data/{name}_test.jsonl")
        self.log_path = config.get("log_path", f"benchmark/data/results/{name.upper()}")
        
        # Create log directory
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize run evaluator
        self.run_evaluator = RunEvaluator()
        
        # Constants
        self.PASS = "PASS"
        self.FAIL = "FAIL"
        
    class TimeoutError(Exception):
        """Timeout error for code execution"""
        pass
        
    def run_with_timeout(self, func, timeout=15):
        """Run function with timeout"""
        result = []
        stop_event = threading.Event()
        
        def target():
            try:
                result.append(func())
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()
            
        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)
        
        if is_timeout:
            raise self.TimeoutError("Function execution timed out")
            
        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]
        
    def check_solution(self, solution: str, test: str, entry_point: str) -> Tuple[str, str]:
        """Check if solution passes the test"""
        try:
            # Clean and validate code
            solution = sanitize(code=solution, entrypoint=entry_point)
            
            # Create test environment with common imports
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }
            
            # Execute solution
            exec(solution, global_dict)
            
            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")
            
            # Execute test
            exec(test, global_dict)
            check = global_dict["check"]
            
            # Run test with timeout
            result = self.run_with_timeout(check, 15)
            
            if result is None:
                return (self.PASS, "The solution passed all test cases.")
                
        except self.TimeoutError:
            return (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations."
            )
        except Exception as e:
            error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
            
            # Log error
            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")
                
            return (self.FAIL, error_message)
            
    def extract_code(self, text: str) -> str:
        """Extract code from text"""
        # For MBPP, we expect the code to be a complete function
        return sanitize(text)
        
    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        """Calculate score for the prediction"""
        # For MBPP, scoring is done in check_solution
        return 0.0, prediction
        
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: float) -> Run:
        """Create a LangSmith run for evaluation"""
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["prompt"]},
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
        
        # Check solution
        result, message = self.check_solution(
            extracted_answer, 
            problem["test"], 
            problem["entry_point"]
        )
        
        # Calculate score
        score = 1.0 if result == self.PASS else 0.0
        
        # Create LangSmith run
        run = self.create_run(problem, final_answer, extracted_answer, score)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)
        
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
            "message": message,
            "run_evaluation": run_evaluation,
        }