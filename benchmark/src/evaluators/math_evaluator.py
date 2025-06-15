"""
Math Evaluator

This module provides a standalone evaluator for mathematical problems using Math-Verify.
"""

import time
from typing import Dict, Any
from pathlib import Path

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from benchmark.src.evaluators.base_evaluator import BaseEvaluator
from benchmark.src.evaluators.registry import register_benchmark
from benchmark.src.evaluators.utils.math_equal import calculate_score

@register_benchmark(
    name="math",
    normalization_keys={
        "id": "id",
        "problem": "problem",
        "solution": "solution",
    }
)
class MathEvaluator(BaseEvaluator):
    """
    Math Evaluator for evaluating math problems using Math-Verify.
    
    This evaluator uses Math-Verify for robust mathematical expression evaluation,
    supporting LaTeX expressions, boxed answers, and various mathematical formats.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the Math Evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Create log directory if it doesn't exist
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize run evaluator for LangSmith compatibility
        self.run_evaluator = RunEvaluator()
    
    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)
    
    def extract_final_answer(self, messages: list) -> str:
        """
        Extract the final answer from a list of messages.
        
        Args:
            messages: List of messages from the agent conversation
            
        Returns:
            The extracted final answer
        """
        final_answer = ""
        
        if not messages:
            return final_answer
            
        last_msg = messages[-1]
        if isinstance(last_msg, tuple) and len(last_msg) > 1:
            final_answer = last_msg[1]
        elif hasattr(last_msg, "content"):
            final_answer = last_msg.content
        elif isinstance(last_msg, dict) and "content" in last_msg:
            final_answer = last_msg["content"]
        
        return final_answer
    
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: int) -> Run:
        """
        Create a LangSmith run for evaluation.
        
        Args:
            problem: The problem dictionary
            final_answer: The raw final answer from the model
            extracted_answer: The extracted answer
            score: The score (0 or 1)
            
        Returns:
            A LangSmith Run object
        """
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["solution"],
                "score": score,
                "passed": score == 1,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
    
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.
        
        Args:
            problem: The problem dictionary with "problem" and "solution" keys
            run_result: The result from running the agent system, including messages
            
        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer from messages
        all_messages = run_result.get("messages", [])
        final_answer = self.extract_final_answer(all_messages)
        
        # Use the enhanced calculate_score method from utils
        score, extracted_answer = calculate_score(problem["solution"], final_answer)
        
        # # Create LangSmith run for evaluation
        # run = self.create_run(problem, final_answer, extracted_answer, score)
        # self.run_evaluator.evaluate_run(run=run)
        
        # Return evaluation results
        return {
            "final_answer": final_answer,
            "score": score,
            "extracted_answer": extracted_answer
        }