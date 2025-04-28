"""
DROP Evaluator

This module provides a standalone evaluator for DROP (Discrete Reasoning Over Paragraphs) problems.
"""

import re
import string
import time
from collections import Counter
from typing import Dict, Any, Tuple, List
from pathlib import Path

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

class DROPEvaluator:
    """Evaluator for DROP problems"""
    
    def __init__(self, name: str = "drop", config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        
        # Set up paths
        self.data_path = config.get("data_path", f"benchmark/data/{name}_test.jsonl")
        self.log_path = config.get("log_path", f"benchmark/data/results/{name.upper()}")
        
        # Create log directory
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize run evaluator
        self.run_evaluator = RunEvaluator()
        
    def normalize_answer(self, s: str) -> List[str]:
        """
        Normalize answers for evaluation.
        
        Args:
            s: The answer string to normalize
            
        Returns:
            List of normalized tokens
        """
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)
            
        def white_space_fix(text):
            return " ".join(text.split())
            
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
            
        def lower(text):
            return text.lower()
            
        return white_space_fix(remove_articles(remove_punc(lower(s))))
        
    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        
        Args:
            ground_truth: The ground truth answer
            prediction: The predicted answer
            
        Returns:
            Tuple of (f1_score, prediction)
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0, prediction
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1, prediction
        
    def create_run(self, problem: Dict[str, Any], final_answer: str, extracted_answer: str, score: float) -> Run:
        """Create a LangSmith run for evaluation"""
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"context": problem["context"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["ref_text"],
                "score": score,
                "passed": score >= 0.3,  # DROP uses 0.3 as threshold
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
        
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.
        
        Args:
            problem: The problem dictionary with "context" and "ref_text" keys
            run_result: The result from running the agent system
            
        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")
        
        # Get reference answers
        ref_answers = problem["ref_text"].split("|")
        
        # Calculate F1 scores for each reference answer
        f1_scores = []
        for ref_answer in ref_answers:
            if ref_answer.strip():
                # Split prediction into parts if it contains multiple answers
                pred_parts = final_answer.split("|")
                for pred_part in pred_parts:
                    f1_score, _ = self.calculate_score(ref_answer, pred_part)
                    f1_scores.append(f1_score)
        
        # Get the best F1 score
        best_score = max(f1_scores) if f1_scores else 0.0
        
        # Create LangSmith run
        run = self.create_run(problem, final_answer, final_answer, best_score)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)
        
        # Log mismatch if score is too low
        if best_score < 0.3:
            with open(f"{self.log_path}/mismatches.log", "a") as f:
                f.write(f"\nContext: {problem['context']}\n")
                f.write(f"Expected: {problem['ref_text']}\n")
                f.write(f"Predicted: {final_answer}\n")
                f.write(f"Score: {best_score}\n")
        
        return {
            "final_answer": final_answer,
            "extracted_answer": final_answer,
            "score": best_score,
            "run_evaluation": run_evaluation,
        }