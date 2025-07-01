import json
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


def _get_sorted_json_files(directory_path: str) -> List[str]:
    """
    Get sorted list of JSON files from the directory.
    
    Args:
        directory_path: Path to the directory containing JSON files
        
    Returns:
        List of sorted JSON file paths
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist.")
        return []
    
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    json_files.sort()
    
    return [os.path.join(directory_path, f) for f in json_files]


def _load_json_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load JSON data from file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Loaded JSON data or None if error
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading {file_path}: {e}")
        return None


def _format_agent_responses(responses: List[Dict[str, Any]]) -> str:
    """
    Format agent responses for analysis.
    
    Args:
        responses: List of agent response dictionaries
        
    Returns:
        Formatted string of agent responses
    """
    formatted_responses = []
    
    for i, response in enumerate(responses, 1):
        agent_id = response.get('agent_id', 'Unknown')
        content = response.get('content', '')
        timestamp = response.get('timestamp', '')
        
        formatted_response = f"Step {i}: Agent {agent_id}\n"
        formatted_response += f"Timestamp: {timestamp}\n"
        formatted_response += f"Content: {content}\n"
        formatted_response += "-" * 50 + "\n"
        
        formatted_responses.append(formatted_response)
    
    return "\n".join(formatted_responses)


def _run_local_generation(model_obj: Union[pipeline, Tuple[AutoModelForCausalLM, AutoTokenizer]], 
                         prompt: str, model_family: str, max_new_tokens: int = 1024) -> Optional[str]:
    """
    Run text generation using local model.
    
    Args:
        model_obj: Model object (pipeline for Llama, tuple for Qwen)
        prompt: Input prompt
        model_family: Model family ('llama' or 'qwen')
        max_new_tokens: Maximum new tokens to generate
        
    Returns:
        Generated text or None if error
    """
    try:
        if model_family == 'llama':
            # For Llama models using pipeline
            messages = [
                {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
                {"role": "user", "content": prompt}
            ]
            
            outputs = model_obj(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )
            
            return outputs[0]["generated_text"][-1]["content"]
            
        elif model_family == 'qwen':
            # For Qwen models using model and tokenizer
            model, tokenizer = model_obj
            
            messages = [
                {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
                {"role": "user", "content": prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
            
        else:
            print(f"Unsupported model family: {model_family}")
            return None
            
    except Exception as e:
        print(f"Error during local generation: {e}")
        return None


def analyze_all_at_once_local(model_obj: Union[pipeline, Tuple[AutoModelForCausalLM, AutoTokenizer]], 
                              directory_path: str, model_family: str):
    """
    Analyze all agent responses at once using local model.
    
    Args:
        model_obj: Local model object
        directory_path: Path to directory containing agent response JSON files
        model_family: Model family ('llama' or 'qwen')
    """
    json_files = _get_sorted_json_files(directory_path)
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Processing {len(json_files)} files with local model...")
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"\n=== File: {filename} ===")
        
        data = _load_json_data(file_path)
        if not data:
            continue
        
        responses = data.get('responses', [])
        if not responses:
            print("No responses found in this file.")
            continue
        
        # Extract problem information if available
        problem_id = data.get('problem_id', 'Unknown')
        agent_system = data.get('agent_system', 'Unknown')
        
        # Format the conversation history
        conversation_history = _format_agent_responses(responses)
        
        # Create the analysis prompt
        prompt = f"""
You are an expert in multi-agent system analysis. Your task is to analyze the following conversation history from a multi-agent system and identify if there are any failures or errors in task execution.

Problem ID: {problem_id}
Agent System: {agent_system}

Please analyze the conversation step by step and identify:
1. Which agent (if any) made an error
2. At which step the error occurred
3. What type of error it was (reasoning error, calculation error, communication error, etc.)
4. The specific reason for the failure

Conversation History:
{conversation_history}

Please provide your analysis in the following format:
Error Agent: [Agent ID or "No Error"]
Error Step: [Step number or "No Error"]
Error Type: [Type of error or "No Error"]
Reason: [Detailed explanation of the error or "No error detected"]

If multiple errors are found, focus on the most critical one that led to task failure.
Note: Focus on identifying clear errors that would lead to incorrect solutions or task failures. Avoid being overly critical of minor issues.
"""
        
        response = _run_local_generation(model_obj, prompt, model_family)
        
        if response:
            print(response)
        else:
            print("Failed to get response from the local model.")
        
        print("\n" + "=" * 80)


def analyze_step_by_step_local(model_obj: Union[pipeline, Tuple[AutoModelForCausalLM, AutoTokenizer]], 
                               directory_path: str, model_family: str):
    """
    Analyze agent responses step by step using local model.
    
    Args:
        model_obj: Local model object
        directory_path: Path to directory containing agent response JSON files
        model_family: Model family ('llama' or 'qwen')
    """
    json_files = _get_sorted_json_files(directory_path)
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Processing {len(json_files)} files with local model...")
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"\n=== File: {filename} ===")
        
        data = _load_json_data(file_path)
        if not data:
            continue
        
        responses = data.get('responses', [])
        if not responses:
            print("No responses found in this file.")
            continue
        
        # Extract problem information if available
        problem_id = data.get('problem_id', 'Unknown')
        agent_system = data.get('agent_system', 'Unknown')
        
        # Analyze each step incrementally
        conversation_so_far = ""
        error_found = False
        
        for i, response in enumerate(responses, 1):
            agent_id = response.get('agent_id', 'Unknown')
            content = response.get('content', '')
            timestamp = response.get('timestamp', '')
            
            step_info = f"Step {i}: Agent {agent_id}\nTimestamp: {timestamp}\nContent: {content}\n" + "-" * 50 + "\n"
            conversation_so_far += step_info
            
            # Ask if there's an error at this step
            prompt = f"""
You are analyzing a multi-agent conversation step by step.

Problem ID: {problem_id}
Agent System: {agent_system}

Here is the conversation history up to step {i}:

{conversation_so_far}

Question: Is there an error or failure in the most recent step (Step {i}) by Agent {agent_id}? 

Please answer with:
- "YES" if there is an error in this step
- "NO" if this step is correct

If YES, also provide:
Error Type: [Type of error]
Reason: [Brief explanation]

If NO, just respond with "NO".
Note: Focus on identifying clear errors that could lead to task failure.
"""
            
            response_text = _run_local_generation(model_obj, prompt, model_family, max_new_tokens=512)
            
            if response_text and "YES" in response_text.upper():
                print(f"Error detected at Step {i} by Agent {agent_id}")
                print(f"Analysis: {response_text}")
                error_found = True
                break
            elif response_text:
                print(f"Step {i}: No error detected")
        
        if not error_found:
            print("No errors detected in the entire conversation.")
        
        print("\n" + "=" * 80)


def _find_error_in_segment_recursive_local(model_obj: Union[pipeline, Tuple[AutoModelForCausalLM, AutoTokenizer]], 
                                           model_family: str, responses: List[Dict[str, Any]], 
                                           start_idx: int, end_idx: int, problem_id: str = 'Unknown', 
                                           agent_system: str = 'Unknown') -> Optional[Tuple[int, str]]:
    """
    Recursively find error in conversation segment using binary search with local model.
    
    Args:
        model_obj: Local model object
        model_family: Model family ('llama' or 'qwen')
        responses: List of response dictionaries
        start_idx: Start index in responses list
        end_idx: End index in responses list
        
    Returns:
        Tuple of (error_step, error_description) or None if no error
    """
    if start_idx > end_idx:
        return None
    
    if start_idx == end_idx:
        # Single step, check if it has an error
        response = responses[start_idx]
        agent_id = response.get('agent_id', 'Unknown')
        content = response.get('content', '')
        
        prompt = f"""
Analyze this single step from a multi-agent conversation:

Problem ID: {problem_id}
Agent System: {agent_system}

Step {start_idx + 1}: Agent {agent_id}
Content: {content}

Is there an error in this step? Respond with:
- "ERROR" if there is an error, followed by the error description
- "NO ERROR" if this step is correct
Note: Focus on identifying clear errors that could lead to task failure.
"""
        
        response_text = _run_local_generation(model_obj, prompt, model_family, max_new_tokens=512)
        
        if response_text and "ERROR" in response_text.upper() and "NO ERROR" not in response_text.upper():
            return (start_idx + 1, response_text)
        else:
            return None
    
    # Format the segment
    segment_responses = responses[start_idx:end_idx + 1]
    conversation_segment = _format_agent_responses(segment_responses)
    
    mid_point = (start_idx + end_idx) // 2
    
    prompt = f"""
You are analyzing a multi-agent conversation to locate errors using binary search.

Problem ID: {problem_id}
Agent System: {agent_system}

Conversation segment (Steps {start_idx + 1} to {end_idx + 1}):
{conversation_segment}

This segment contains {end_idx - start_idx + 1} steps. The middle point is around step {mid_point + 1}.

Question: If there is an error in this segment, is it more likely in:
A) The UPPER HALF (steps {start_idx + 1} to {mid_point + 1})
B) The LOWER HALF (steps {mid_point + 2} to {end_idx + 1})

Please respond with exactly:
- "UPPER HALF" if the error is in the first half
- "LOWER HALF" if the error is in the second half  
- "NO ERROR" if no error is detected in this segment

If you detect an error, also provide a brief reason.
Note: Focus on identifying clear errors that could lead to task failure.
"""
    
    response_text = _run_local_generation(model_obj, prompt, model_family, max_new_tokens=512)
    
    if not response_text:
        return None
    
    mid_idx = (start_idx + end_idx) // 2
    
    if "UPPER HALF" in response_text.upper():
        return _find_error_in_segment_recursive_local(model_obj, model_family, responses, start_idx, mid_idx, problem_id, agent_system)
    elif "LOWER HALF" in response_text.upper():
        return _find_error_in_segment_recursive_local(model_obj, model_family, responses, mid_idx + 1, end_idx, problem_id, agent_system)
    else:
        return None


def analyze_binary_search_local(model_obj: Union[pipeline, Tuple[AutoModelForCausalLM, AutoTokenizer]], 
                                directory_path: str, model_family: str):
    """
    Analyze agent responses using binary search with local model.
    
    Args:
        model_obj: Local model object
        directory_path: Path to directory containing agent response JSON files
        model_family: Model family ('llama' or 'qwen')
    """
    json_files = _get_sorted_json_files(directory_path)
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Processing {len(json_files)} files with local model...")
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        print(f"\n=== File: {filename} ===")
        
        data = _load_json_data(file_path)
        if not data:
            continue
        
        responses = data.get('responses', [])
        if not responses:
            print("No responses found in this file.")
            continue
        
        # Extract problem information if available
        problem_id = data.get('problem_id', 'Unknown')
        agent_system = data.get('agent_system', 'Unknown')
        
        if len(responses) == 1:
            # Only one step, analyze directly
            response = responses[0]
            agent_id = response.get('agent_id', 'Unknown')
            content = response.get('content', '')
            
            prompt = f"""
Analyze this single-step conversation:

Problem ID: {problem_id}
Agent System: {agent_system}

Agent {agent_id}: {content}

Is there an error? Provide analysis in format:
Error Agent: [Agent ID or "No Error"]
Error Step: [1 or "No Error"]
Reason: [Explanation]
Note: Focus on identifying clear errors that could lead to task failure.
"""
            
            response_text = _run_local_generation(model_obj, prompt, model_family)
            if response_text:
                print(response_text)
        else:
            # Multiple steps, use binary search
            result = _find_error_in_segment_recursive_local(model_obj, model_family, responses, 0, len(responses) - 1, problem_id, agent_system)
            
            if result:
                error_step, error_description = result
                if error_step <= len(responses):
                    error_response = responses[error_step - 1]
                    agent_id = error_response.get('agent_id', 'Unknown')
                    
                    print(f"Error found at Step {error_step} by Agent {agent_id}")
                    print(f"Error Description: {error_description}")
                    print(f"Error Agent: {agent_id}")
                    print(f"Error Step: {error_step}")
                else:
                    print(f"Error reported at step {error_step}, but step is out of range.")
            else:
                print("No errors detected using binary search.")
        
        print("\n" + "=" * 80)