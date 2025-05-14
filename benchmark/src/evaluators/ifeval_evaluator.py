"""
IFEval Evaluator

This module provides a standalone evaluator for instruction following evaluation.
"""

import json
import collections
from typing import Dict, Any, List
from pathlib import Path

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from benchmark.src.evaluators.utils.ifeval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose,
    InputExample,
    OutputExample,
)

class IFEvalEvaluator:
    """
    Evaluator for instruction following tasks.
    Integrates the official IFEval implementation.
    """
    
    def __init__(self, name: str = "ifeval", config: Dict[str, Any] = None):
        """
        Initialize the IFEval Evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.run_evaluator = RunEvaluator()
        
    def preprocess_input(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the input problem before passing to the agent system.
        
        Args:
            problem: Raw problem dictionary
            
        Returns:
            Preprocessed problem dictionary
        """
        # Create input example
        input_example = InputExample(
            key=problem.get("key", 0),
            instruction_id_list=problem["instruction_id_list"],
            prompt=problem["prompt"],
            kwargs=problem["kwargs"]
        )
        
        # Return preprocessed input
        return {
            "problem": input_example.prompt,  # Use the prompt as the main problem
            "instruction_id_list": input_example.instruction_id_list,
            "kwargs": input_example.kwargs,
            "key": input_example.key,
            "original_problem": problem  # Keep original data for evaluation
        }
        
    def calculate_metrics(self, output: OutputExample) -> Dict[str, Any]:
        """Calculate metrics for a single example using official logic"""
        follow_instruction_list = output.follow_instruction_list
        instruction_id_list = output.instruction_id_list
        
        # Calculate instruction-level metrics
        instruction_total = len(instruction_id_list)
        instruction_correct = sum(follow_instruction_list)
        instruction_accuracy = instruction_correct / instruction_total if instruction_total > 0 else 0
        
        # Calculate instruction type metrics
        type_metrics = collections.defaultdict(lambda: {"total": 0, "correct": 0})
        for instruction_id, followed in zip(instruction_id_list, follow_instruction_list):
            instr_type = instruction_id.split(":")[0]  # e.g., "keywords" from "keywords:existence"
            type_metrics[instr_type]["total"] += 1
            if followed:
                type_metrics[instr_type]["correct"] += 1
        
        # Convert type metrics to accuracies
        type_accuracies = {}
        for instr_type, counts in type_metrics.items():
            accuracy = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
            type_accuracies[instr_type] = {
                "accuracy": accuracy,
                "correct": counts["correct"],
                "total": counts["total"]
            }
        
        return {
            "instruction_accuracy": instruction_accuracy,
            "instruction_correct": instruction_correct,
            "instruction_total": instruction_total,
            "all_instructions_followed": output.follow_all_instructions,
            "instruction_results": output.follow_instruction_list,
            "type_accuracies": type_accuracies
        }
        
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response.
        
        Args:
            problem: The problem dictionary with "prompt", "instruction_id_list", and "kwargs"
            run_result: The result from running the agent system
            
        Returns:
            Evaluation results dictionary
        """
        # Extract the final answer
        final_answer = run_result.get("final_answer", "")
        
        # Get original problem data
        original_problem = problem.get("original_problem", problem)
        
        # Create input example for official evaluator
        input_example = InputExample(
            key=original_problem.get("key", 0),
            instruction_id_list=original_problem["instruction_id_list"],
            prompt=original_problem["prompt"],
            kwargs=original_problem["kwargs"]
        )
        
        # Create response dictionary
        prompt_to_response = {input_example.prompt: final_answer}
        
        # Run both strict and loose evaluations
        strict_result = test_instruction_following_strict(input_example, prompt_to_response)
        loose_result = test_instruction_following_loose(input_example, prompt_to_response)
        
        # Calculate metrics
        strict_metrics = self.calculate_metrics(strict_result)
        loose_metrics = self.calculate_metrics(loose_result)
        
        # Use strict accuracy as the main score for metrics collection
        score = strict_metrics["instruction_accuracy"]
        
        evaluation_result = {
            "final_answer": final_answer,
            "score": score,  # This is what supervisor_mas will use
            "strict_evaluation": strict_metrics,
            "loose_evaluation": loose_metrics
        }
        
        return evaluation_result