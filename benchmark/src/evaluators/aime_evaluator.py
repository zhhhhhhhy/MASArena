"""
AIME Evaluator

Standalone evaluator for AIME-style math problems using Math-Verify for robust mathematical expression evaluation.
"""

import re
import time
from typing import Dict, Any, Tuple
from pathlib import Path
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from benchmark.src.evaluators.base_evaluator import BaseEvaluator
from benchmark.src.evaluators.registry import register_benchmark
from benchmark.src.evaluators.utils.math_equal import calculate_score

@register_benchmark(
    name="aime",
    normalization_keys={
        "problem": "question",
        "solution": "answer",
    }
)
class AIMEEvaluator(BaseEvaluator):
    """
    Evaluator for AIME-style math problems.
    Uses Math-Verify for robust mathematical expression evaluation.
    """
    def __init__(self, name: str = "aime", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.run_evaluator = RunEvaluator()

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: int) -> Run:
        import uuid
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"question": problem["problem"]},
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
        final_answer = run_result.get("final_answer", "")
        score, extracted_answer = self.calculate_score(problem["solution"], final_answer)
        run = self.create_run(problem, final_answer, extracted_answer, score)
        self.run_evaluator.evaluate_run(run=run)
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
        }