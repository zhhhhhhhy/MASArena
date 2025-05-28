"""
Utility functions for normalizing benchmark problem data.
"""
from typing import Dict, Any
from .benchmark_key_mappings import BENCHMARK_KEY_MAPPINGS # Import mappings

def normalize_problem_keys(problem: Dict[str, Any], benchmark_name: str, problem_index: int) -> Dict[str, Any]:
    """
    Normalizes the keys of a problem dictionary based on the benchmark name.

    Args:
        problem: The original problem dictionary.
        benchmark_name: The name of the benchmark to get the key mapping for.
        problem_index: The index of the problem, used for generating a default ID.

    Returns:
        A new dictionary with normalized keys.
    
    Raises:
        ValueError: If the benchmark_name is not found in BENCHMARK_KEY_MAPPINGS.
    """
    key_mapping = BENCHMARK_KEY_MAPPINGS.get(benchmark_name)
    if key_mapping is None:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Supported: {', '.join(BENCHMARK_KEY_MAPPINGS.keys())}")

    normalized_problem = {
        "id": problem.get(key_mapping["id"], f"problem_{problem_index + 1}"),
        "problem": problem.get(key_mapping["problem"], ""),
        "solution": problem.get(key_mapping["solution"], ""),
        "test": problem.get(key_mapping["test"], ""),
        "entry_point": problem.get(key_mapping["entry_point"], ""),
    }
    if key_mapping.get("test_imports") is not None: # Check if 'test_imports' key exists and is not None
        normalized_problem["test_imports"] = problem.get(key_mapping["test_imports"], [])
    return normalized_problem 