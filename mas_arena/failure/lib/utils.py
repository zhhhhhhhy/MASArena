import json
import os
from typing import List, Dict, Any, Optional, Tuple
import time
import datetime
from openai import AzureOpenAI, OpenAI


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


def _make_api_call(client, model: str, messages: List[Dict[str, str]], 
                   max_tokens: int = 1024) -> Optional[str]:
    """
    Make API call to OpenAI (Azure or standard).
    
    Args:
        client: OpenAI or AzureOpenAI client
        model: Model name
        messages: List of messages for the conversation
        max_tokens: Maximum tokens for response
        
    Returns:
        Response content or None if error
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error making API call: {e}")
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


def all_at_once(client, directory_path: str, model: str, max_tokens: int = 1024):
    """
    Analyze all agent responses at once to identify failure attribution.
    
    Args:
        client: OpenAI or AzureOpenAI client
        directory_path: Path to directory containing agent response JSON files
        model: Model name to use
        max_tokens: Maximum tokens for response
    """
    json_files = _get_sorted_json_files(directory_path)
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Processing {len(json_files)} files...")
    
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
        
        # Format the conversation history
        conversation_history = _format_agent_responses(responses)
        
        # Extract problem information if available
        problem_id = data.get('problem_id', 'Unknown')
        agent_system = data.get('agent_system', 'Unknown')
        
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
        
        messages = [
            {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
            {"role": "user", "content": prompt}
        ]
        
        response = _make_api_call(client, model, messages, max_tokens)
        
        if response:
            print(response)
        else:
            print("Failed to get response from the model.")
        
        print("\n" + "=" * 80)
        time.sleep(1)  # Rate limiting


def convert_txt_to_json(txt_filepath: str, json_filepath: str, method: str, model: str, directory_path: str) -> None:
    """
    Convert analysis results from txt format to structured JSON format.
    
    Args:
        txt_filepath: Path to the input txt file
        json_filepath: Path to the output json file
        method: Analysis method used
        model: Model used for analysis
        directory_path: Input directory path
    """
    try:
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the content to extract structured information
        lines = content.split('\n')
        
        # Initialize the JSON structure
        result = {
            "metadata": {
                "method": method,
                "model": model,
                "input_directory": directory_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "analysis_type": "failure_attribution"
            },
            "files_analyzed": [],
            "summary": {
                "total_files": 0,
                "files_with_errors": 0,
                "files_without_errors": 0
            }
        }
        
        current_file = None
        current_file_data = None
        
        for line in lines:
            line = line.strip()
            
            # Detect file processing start
            if line.startswith("=== File:") and line.endswith("==="):
                # Save previous file if exists
                if current_file_data:
                    result["files_analyzed"].append(current_file_data)
                
                # Start new file
                filename = line.replace("=== File:", "").replace("===", "").strip()
                current_file_data = {
                    "filename": filename,
                    "analysis_result": {
                        "error_detected": False,
                        "error_agent": None,
                        "error_step": None,
                        "error_type": None,
                        "error_reason": None
                    }
                }
                current_file = filename
            
            # Collect analysis content for current file
            elif current_file_data and line and not line.startswith("==="):
                
                # Parse specific error information
                if line.startswith("Error Agent:"):
                    error_agent = line.replace("Error Agent:", "").strip()
                    if error_agent.lower() not in ["no error", "none", ""]:
                        current_file_data["analysis_result"]["error_detected"] = True
                        current_file_data["analysis_result"]["error_agent"] = error_agent
                
                elif line.startswith("Error Step:"):
                    error_step = line.replace("Error Step:", "").strip()
                    if error_step.lower() not in ["no error", "none", ""]:
                        try:
                            current_file_data["analysis_result"]["error_step"] = int(error_step)
                        except ValueError:
                            current_file_data["analysis_result"]["error_step"] = error_step
                
                elif line.startswith("Error Type:"):
                    error_type = line.replace("Error Type:", "").strip()
                    if error_type.lower() not in ["no error", "none", ""]:
                        current_file_data["analysis_result"]["error_type"] = error_type
                
                elif line.startswith("Reason:") or line.startswith("Error Description:"):
                    reason = line.replace("Reason:", "").replace("Error Description:", "").strip()
                    if reason.lower() not in ["no error", "none", ""]:
                        current_file_data["analysis_result"]["error_reason"] = reason
                
                elif "Error found at Step" in line:
                    current_file_data["analysis_result"]["error_detected"] = True
                    # Extract step number
                    try:
                        step_part = line.split("Step")[1].split("by")[0].strip()
                        current_file_data["analysis_result"]["error_step"] = int(step_part)
                    except (IndexError, ValueError):
                        pass
                
                elif "No errors detected" in line or "No Error" in line:
                    current_file_data["analysis_result"]["error_detected"] = False
        
        # Add the last file if exists
        if current_file_data:
            result["files_analyzed"].append(current_file_data)
        
        # Calculate summary statistics
        result["summary"]["total_files"] = len(result["files_analyzed"])
        result["summary"]["files_with_errors"] = sum(1 for f in result["files_analyzed"] 
                                                      if f["analysis_result"]["error_detected"])
        result["summary"]["files_without_errors"] = (result["summary"]["total_files"] - 
                                                      result["summary"]["files_with_errors"])
        
        # Write JSON file
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"JSON output saved to: {json_filepath}")
        
    except Exception as e:
        print(f"Error converting to JSON: {e}")


def generate_timestamped_filename(base_name: str, extension: str) -> str:
    """
    Generate a filename with timestamp suffix.
    
    Args:
        base_name: Base filename without extension
        extension: File extension (with or without dot)
    
    Returns:
        Filename with timestamp suffix
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not extension.startswith('.'):
        extension = '.' + extension
    return f"{base_name}_{timestamp}{extension}"


def step_by_step(client, directory_path: str, model: str, max_tokens: int = 1024):
    """
    Analyze agent responses step by step to identify failure attribution.
    
    Args:
        client: OpenAI or AzureOpenAI client
        directory_path: Path to directory containing agent response JSON files
        model: Model name to use
        max_tokens: Maximum tokens for response
    """
    json_files = _get_sorted_json_files(directory_path)
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Processing {len(json_files)} files...")
    
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
You are analyzing a multi-agent conversation step by step to identify errors that could lead to task failure.

Problem ID: {problem_id}
Agent System: {agent_system}

Here is the conversation history up to step {i}:

{conversation_so_far}

Question: Is there an error or failure in the most recent step (Step {i}) by Agent {agent_id}? 

Please answer with:
- "YES" if there is an error in this step that could lead to incorrect solutions or task failure
- "NO" if this step is correct or contains only minor issues

If YES, also provide:
Error Type: [Type of error]
Reason: [Brief explanation]

If NO, just respond with "NO".

Note: Focus on identifying clear errors that would derail the problem-solving process. Avoid being overly critical of minor issues or stylistic choices.
"""
            
            messages = [
                {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = _make_api_call(client, model, messages, max_tokens)
            
            if response_text and "YES" in response_text.upper():
                print(f"Error detected at Step {i} by Agent {agent_id}")
                print(f"Analysis: {response_text}")
                error_found = True
                break
            elif response_text:
                print(f"Step {i}: No error detected")
            
            time.sleep(0.5)  # Rate limiting
        
        if not error_found:
            print("No errors detected in the entire conversation.")
        
        print("\n" + "=" * 80)


def _construct_binary_search_prompt(conversation_segment: str, start_step: int, end_step: int, problem_id: str = "Unknown", agent_system: str = "Unknown") -> str:
    """
    Construct prompt for binary search analysis.
    
    Args:
        conversation_segment: The conversation segment to analyze
        start_step: Starting step number
        end_step: Ending step number
        problem_id: Problem identifier
        agent_system: Agent system type
        
    Returns:
        Formatted prompt for binary search
    """
    mid_point = (start_step + end_step) // 2
    
    prompt = f"""
You are analyzing a multi-agent conversation to locate errors using binary search.

Problem ID: {problem_id}
Agent System: {agent_system}

Conversation segment (Steps {start_step} to {end_step}):
{conversation_segment}

This segment contains {end_step - start_step + 1} steps. The middle point is around step {mid_point}.

Question: If there is an error in this segment that could lead to task failure, is it more likely in:
A) The UPPER HALF (steps {start_step} to {mid_point})
B) The LOWER HALF (steps {mid_point + 1} to {end_step})

Please respond with exactly:
- "UPPER HALF" if the error is in the first half
- "LOWER HALF" if the error is in the second half  
- "NO ERROR" if no error is detected in this segment

If you detect an error, also provide a brief reason.

Note: Focus on identifying clear errors that would lead to incorrect solutions or task failures. Avoid being overly critical of minor issues.
"""
    
    return prompt


def _find_error_in_segment_recursive(client, model: str, responses: List[Dict[str, Any]], 
                                    start_idx: int, end_idx: int, max_tokens: int, 
                                    problem_id: str = "Unknown", agent_system: str = "Unknown") -> Optional[Tuple[int, str]]:
    """
    Recursively find error in conversation segment using binary search.
    
    Args:
        client: OpenAI or AzureOpenAI client
        model: Model name
        responses: List of response dictionaries
        start_idx: Start index in responses list
        end_idx: End index in responses list
        max_tokens: Maximum tokens for response
        problem_id: Problem identifier
        agent_system: Agent system type
        
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

Is there an error in this step that could lead to task failure? Respond with:
- "ERROR" if there is an error, followed by the error description
- "NO ERROR" if this step is correct or contains only minor issues

Note: Focus on identifying clear errors that would lead to incorrect solutions or task failures.
"""
        
        messages = [
            {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
            {"role": "user", "content": prompt}
        ]
        
        response_text = _make_api_call(client, model, messages, max_tokens)
        
        if response_text and "ERROR" in response_text.upper() and "NO ERROR" not in response_text.upper():
            return (start_idx + 1, response_text)
        else:
            return None
    
    # Format the segment
    segment_responses = responses[start_idx:end_idx + 1]
    conversation_segment = _format_agent_responses(segment_responses)
    
    prompt = _construct_binary_search_prompt(conversation_segment, start_idx + 1, end_idx + 1, problem_id, agent_system)
    
    messages = [
        {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
        {"role": "user", "content": prompt}
    ]
    
    response_text = _make_api_call(client, model, messages, max_tokens)
    
    if not response_text:
        return None
    
    mid_idx = (start_idx + end_idx) // 2
    
    if "UPPER HALF" in response_text.upper():
        return _find_error_in_segment_recursive(client, model, responses, start_idx, mid_idx, max_tokens, problem_id, agent_system)
    elif "LOWER HALF" in response_text.upper():
        return _find_error_in_segment_recursive(client, model, responses, mid_idx + 1, end_idx, max_tokens, problem_id, agent_system)
    else:
        return None


def _report_binary_search_error(error_step: int, error_description: str, responses: List[Dict[str, Any]]):
    """
    Report the error found through binary search.
    
    Args:
        error_step: Step number where error was found
        error_description: Description of the error
        responses: List of all responses
    """
    if error_step <= len(responses):
        error_response = responses[error_step - 1]
        agent_id = error_response.get('agent_id', 'Unknown')
        
        print(f"Error found at Step {error_step} by Agent {agent_id}")
        print(f"Error Description: {error_description}")
        print(f"Error Agent: {agent_id}")
        print(f"Error Step: {error_step}")
    else:
        print(f"Error reported at step {error_step}, but step is out of range.")


def binary_search(client, directory_path: str, model: str, max_tokens: int = 1024):
    """
    Analyze agent responses using binary search to identify failure attribution.
    
    Args:
        client: OpenAI or AzureOpenAI client
        directory_path: Path to directory containing agent response JSON files
        model: Model name to use
        max_tokens: Maximum tokens for response
    """
    json_files = _get_sorted_json_files(directory_path)
    
    if not json_files:
        print("No JSON files found in the specified directory.")
        return
    
    print(f"Processing {len(json_files)} files...")
    
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

Is there an error that could lead to task failure? Provide analysis in format:
Error Agent: [Agent ID or "No Error"]
Error Step: [1 or "No Error"]
Reason: [Explanation]

Note: Focus on identifying clear errors that would lead to incorrect solutions or task failures.
"""
            
            messages = [
                {"role": "system", "content": "You are an expert analyst for multi-agent systems."},
                {"role": "user", "content": prompt}
            ]
            
            response_text = _make_api_call(client, model, messages, max_tokens)
            if response_text:
                print(response_text)
        else:
            # Multiple steps, use binary search
            result = _find_error_in_segment_recursive(client, model, responses, 0, len(responses) - 1, max_tokens, problem_id, agent_system)
            
            if result:
                error_step, error_description = result
                _report_binary_search_error(error_step, error_description, responses)
            else:
                print("No errors detected using binary search.")
        
        print("\n" + "=" * 80)
        time.sleep(1)  # Rate limiting