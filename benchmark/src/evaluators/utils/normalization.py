"""
Utility functions for normalizing benchmark problem data.
"""
from typing import Dict, Any

def normalize_problem_keys(problem: Dict[str, Any], key_mapping: Dict[str, str], problem_index: int) -> Dict[str, Any]:
    """
    Normalizes the keys of a problem dictionary based on a provided key mapping.

    Args:
        problem: The original problem dictionary.
        key_mapping: A dictionary that maps standard keys ('id', 'problem', 'solution', etc.)
                     to the actual keys in the problem dictionary.
        problem_index: The index of the problem, used for generating a default ID if not present.

    Returns:
        A new dictionary with normalized keys.
    """
    normalized_problem = {}
    
    # Define a mapping from standard internal keys to the keys expected in the source data
    # and whether they are essential.
    key_definitions = {
        "id": "id",
        "problem": "problem",
        "solution": "solution",
        "test": "test",
        "entry_point": "entry_point",
        "test_imports": "test_imports",
        "instruction_id_list": "instruction_id_list",
        "kwargs": "kwargs"
    }

    for standard_key, source_key_name in key_definitions.items():
        source_key = key_mapping.get(source_key_name)
        if source_key and source_key in problem:
            normalized_problem[standard_key] = problem[source_key]

    # Ensure a unique ID for the problem, generating one if not provided.
    if "id" not in normalized_problem:
        normalized_problem["id"] = f"problem_{problem_index + 1}"
        
    return normalized_problem 