"""
AIME Evaluator

Standalone evaluator for AIME-style math problems.
"""

import re
import time
from typing import Dict, Any, Tuple

from benchmark.src.evaluators.base_evaluator import BaseEvaluator
from benchmark.src.evaluators.registry import register_benchmark

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
    Extracts answers and compares with expected answers (numeric/string match).
    """
    def __init__(self, name: str = "aime", config: Dict[str, Any] = None):
        super().__init__(name, config)

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] = None):
        return cls(name, config)

    def extract_answer(self, text: str) -> str:
        """
        Extract the answer from model output text (last number or string).
        """
        # Try to extract the last number (int/float)
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            return matches[-1].replace(",", "").strip()
        # Fallback: last non-empty line
        lines = [line.strip() for line in str(text).splitlines() if line.strip()]
        return lines[-1] if lines else str(text).strip()

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected = self.extract_answer(expected_output)
        pred = self.extract_answer(prediction)
        # Try numeric comparison
        try:
            if float(pred) == float(expected):
                return 1, pred
        except Exception:
            pass
        # Fallback: string match (ignore whitespace)
        if str(pred).strip() == str(expected).strip():
            return 1, pred
        return 0, pred

 

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        final_answer = run_result.get("final_answer", "")
        score, extracted_answer = self.calculate_score(problem["solution"], final_answer)
        return {
            "final_answer": final_answer,
            "extracted_answer": extracted_answer,
            "score": score,
        } 