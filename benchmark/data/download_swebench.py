#!/usr/bin/env python3
"""
Download and process SWE-bench dataset

This script downloads the SWE-bench dataset from Hugging Face and converts it to the
format expected by the benchmark framework.
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset


def extract_repo_info(repo_string):
    """
    Extract repository owner and name from a repo string.
    
    Args:
        repo_string: String in format like "owner/repo"
        
    Returns:
        Tuple of (repo_owner, repo_name)
    """
    if not repo_string or '/' not in repo_string:
        return "", ""
    
    parts = repo_string.split('/')
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def extract_test_commands(problem_statement, fail_to_pass, pass_to_pass):
    """
    Extract test commands from problem statement and test information.
    
    Args:
        problem_statement: The problem description
        fail_to_pass: FAIL_TO_PASS information
        pass_to_pass: PASS_TO_PASS information
        
    Returns:
        List of test commands
    """
    commands = []
    
    # Try to extract test commands from problem statement
    test_patterns = [
        r"run the test(?:s)? with[:\s]+`([^`]+)`",
        r"test(?:s)? can be run with[:\s]+`([^`]+)`",
        r"run[:\s]+`([^`]*test[^`]*)`",
        r"execute[:\s]+`([^`]*test[^`]*)`"
    ]
    
    for pattern in test_patterns:
        matches = re.findall(pattern, problem_statement, re.IGNORECASE)
        if matches:
            commands.extend(matches)
    
    # If no test commands found in problem statement, check FAIL_TO_PASS and PASS_TO_PASS
    if not commands:
        if fail_to_pass:
            commands.append(fail_to_pass)
        if pass_to_pass:
            commands.append(pass_to_pass)
    
    # Remove duplicates while preserving order
    unique_commands = []
    for cmd in commands:
        if cmd and cmd not in unique_commands:
            unique_commands.append(cmd)
    
    return unique_commands


def extract_files_to_edit(patch):
    """
    Extract files to be edited from patch.
    
    Args:
        patch: The patch/diff string
        
    Returns:
        List of file paths that need to be edited
    """
    if not patch:
        return []
    
    # Look for file paths in diff headers
    files = []
    diff_file_patterns = [
        r"diff --git a/(.*?) b/",
        r"\+\+\+ b/(.*)",
        r"--- a/(.*)"
    ]
    
    for pattern in diff_file_patterns:
        matches = re.findall(pattern, patch)
        files.extend(matches)
    
    # Remove duplicates while preserving order
    unique_files = []
    for file in files:
        if file and file not in unique_files:
            unique_files.append(file)
    
    return unique_files


def download_swebench(output_dir="data", split="dev", lite=True):
    """
    Download SWE-bench dataset and convert to the benchmark format.
    
    Args:
        output_dir: Directory to save the processed data
        split: Dataset split to download (dev, test)
        lite: Whether to use SWE-bench Lite (smaller version) or full dataset
    
    Returns:
        Path to the processed dataset file
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset to load
    dataset_name = "princeton-nlp/SWE-bench_Lite" if lite else "princeton-nlp/SWE-bench"
    print(f"Loading {dataset_name} dataset ({split} split)...")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} examples")
    
    # Define output path
    prefix = "swebench_lite" if lite else "swebench"
    output_path = output_dir / f"{prefix}_{split}.jsonl"
    
    # Convert and save the dataset
    print(f"Converting and saving to {output_path}...")
    with open(output_path, "w") as f:
        for i, item in enumerate(tqdm(dataset)):
            # Extract repo owner and name from repo field
            repo_owner, repo_name = extract_repo_info(item.get("repo", ""))
            
            # Extract test commands
            test_commands = extract_test_commands(
                item.get("problem_statement", ""),
                item.get("FAIL_TO_PASS", ""),
                item.get("PASS_TO_PASS", "")
            )
            
            # Extract files to edit from patch
            files_to_edit = extract_files_to_edit(item.get("patch", ""))
            
            # Generate a unique problem ID if instance_id is not available
            problem_id = item.get("instance_id", f"swebench_{split}_{i}")
            
            # Create repo_url from repo_owner and repo_name
            repo_url = f"https://github.com/{repo_owner}/{repo_name}" if repo_owner and repo_name else "https://github.com/dummy/repo"
            
            # Get a test command to use (use the first one if available, otherwise a default)
            test_command = test_commands[0] if test_commands else "echo 'Test passed'"
            
            # Map from HF dataset format to our benchmark format
            processed_item = {
                "problem_id": problem_id,
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "repo_url": repo_url,  # Add repo_url field
                "base_commit": item.get("base_commit", ""),
                "problem": item.get("problem_statement", ""),  # Main problem description
                "issue_title": f"Fix issue in {repo_owner}/{repo_name}",  # Generate a title
                "issue_body": item.get("problem_statement", ""),
                "pr_diff": item.get("patch", ""),  # Original solution for reference
                "solution": item.get("patch", ""),  # For compatibility with other benchmarks
                "test_command": test_command,  # Add test_command field
                "test_commands": test_commands,
                "files_to_edit": files_to_edit,
                "metadata": {
                    "hints_text": item.get("hints_text", ""),
                    "created_at": item.get("created_at", ""),
                    "version": item.get("version", ""),
                    "test_patch": item.get("test_patch", ""),
                    "environment_setup_commit": item.get("environment_setup_commit", "")
                }
            }
            
            # Write to JSONL file
            f.write(json.dumps(processed_item) + "\n")
    
    print(f"Dataset saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download and process SWE-bench dataset")
    parser.add_argument("--output-dir", type=str, default="data", 
                        help="Directory to save the processed data")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"],
                        help="Dataset split to download")
    parser.add_argument("--full", action="store_true", 
                        help="Use full SWE-bench dataset instead of Lite version")
    args = parser.parse_args()
    
    # Download and process the dataset
    download_swebench(
        output_dir=args.output_dir, 
        split=args.split, 
        lite=not args.full
    )


if __name__ == "__main__":
    main() 