from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from typing import Tuple

def calculate_score(expected_output: str, prediction: str) -> Tuple[int, str]:
    """
    Calculate score using Math-Verify's verification capabilities.
    For math problems, gold answers can be various formats, but predictions may contain full explanations.
    """
    try:
        # Parse the gold answer
        gold_parsed = parse(
            expected_output,
            extraction_mode="first_match",
        )
        
        if len(gold_parsed) != 0:
            # Parse the model's answer with optimized LaTeX requirements
            answer_parsed = parse(
                prediction,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=True,  # Allow extraction without LaTeX anchors
                    )
                ],
                extraction_mode="first_match",
            )
            
            if len(answer_parsed) != 0:
                try:
                    # Verify the answers match
                    is_correct = verify(gold_parsed, answer_parsed)
                    # Extract the actual value from the parsed result
                    extracted_value = str(answer_parsed[0]) if isinstance(answer_parsed, list) else str(answer_parsed)
                    return int(is_correct), extracted_value
                except Exception as e:
                    print(f"Verification failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
    except Exception as e:
        print(f"Parsing failed: {e}")
    
    # Enhanced fallback: try to extract numbers from text
    import re
    
    # Look for numbers that might be answers
    number_patterns = [
        r"(?:final answer|answer|result|solution)[\s:]*(\d+)",  # "final answer is 81"
        r"(\d+)\.?\s*$",  # number at end of text
        r"\b(\d+)\b"  # any standalone number
    ]
    
    for pattern in number_patterns:
        matches = re.findall(pattern, prediction, re.IGNORECASE)
        if matches:
            candidate = matches[-1]  # Take the last match
            try:
                if int(candidate) == int(expected_output):
                    return 1, candidate
            except (ValueError, TypeError):
                continue
    
    # Final fallback: simple string comparison
    expected = str(expected_output).strip()
    pred = str(prediction).strip()
    if expected == pred:
        return 1, pred
    return 0, pred