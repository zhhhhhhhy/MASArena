"""
SWE-bench Evaluator

This module provides an evaluator for SWE-bench problems, supporting different agent system output formats.
"""

import json
import os
import re
import time
import tempfile
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from threading import Thread

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from benchmark.src.evaluators.utils.sanitize import code_extract


class SWEBenchEvaluator:
    """
    Evaluator for SWE-bench problems
    
    This evaluator supports different agent system output formats and abstracts away the 
    implementation details of applying/testing patches to repositories.
    """
    
    def __init__(self, name: str = "swebench", config: Dict[str, Any] = None):
        """
        Initialize the SWE-bench evaluator
        
        Args:
            name: Name of the evaluator
            config: Configuration options including:
                - data_path: Path to the test data file
                - log_path: Path to save logs and results
                - repos_path: Path to store git repositories
                - timeout: Timeout for patch application and testing (seconds)
                - verbose: Enable verbose logging
                - use_mcp: Use MCP servers for evaluation
                - mcp_executable: Path to MCP server executables
        """
        self.name = name
        self.config = config or {}
        
        # Set up paths
        self.data_path = config.get("data_path", f"benchmark/data/{name}_test.jsonl")
        self.log_path = config.get("log_path", f"benchmark/data/results/{name.upper()}")
        self.repos_path = config.get("repos_path", "benchmark/data/repos")
        
        # Setup timeout and other configs
        self.timeout = config.get("timeout", 600)  # 10 minutes default timeout
        self.verbose = config.get("verbose", False)
        
        # MCP server settings
        self.use_mcp = config.get("use_mcp", True)
        self.mcp_path = config.get("mcp_executable", "benchmark/mcp_servers")
        
        # Create directories
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        Path(self.repos_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize run evaluator
        self.run_evaluator = RunEvaluator()
    
    class TimeoutError(Exception):
        """Timeout error for code execution"""
        pass
    
    def run_with_timeout(self, func, args=None, kwargs=None, timeout=None):
        """Run function with timeout"""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if timeout is None:
            timeout = self.timeout
            
        result = [None]
        error = [None]
        completed = [False]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
                completed[0] = True
            except Exception as e:
                error[0] = e
                if self.verbose:
                    traceback.print_exc()
            
        thread = Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if not completed[0]:
            if thread.is_alive():
                raise self.TimeoutError(f"Function execution timed out after {timeout} seconds")
            elif error[0]:
                raise error[0]
            
        return result[0]
    
    def _run_mcp_command(self, server_type: str, command: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a command using an MCP server
        
        Args:
            server_type: Type of MCP server (git, filesystem, process, evaluator)
            command: Command to run
            args: Arguments for the command
        
        Returns:
            Response from the MCP server
        """
        if not self.use_mcp:
            raise ValueError("MCP servers are not enabled in configuration")
            
        server_map = {
            "git": "git_server.py",
            "fs": "filesystem_server.py",
            "process": "process_server.py",
            "env": "environment_server.py",
            "evaluator": "evaluator_server.py"
        }
        
        if server_type not in server_map:
            raise ValueError(f"Unknown MCP server type: {server_type}")
        
        server_script = os.path.join(self.mcp_path, server_map[server_type])
        if not os.path.exists(server_script):
            raise FileNotFoundError(f"MCP server script not found: {server_script}")
        
        # Prepare the command request
        request = json.dumps({"command": command, "args": args})
        
        # Run the MCP server process
        try:
            process = subprocess.Popen(
                ["python3", server_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            
            # Send the request and get the response
            stdout, stderr = process.communicate(input=request + "\n", timeout=self.timeout)
            
            if stderr and self.verbose:
                print(f"MCP Server stderr: {stderr}")
                
            if not stdout:
                raise RuntimeError(f"No response from MCP server: {server_type}/{command}")
                
            response = json.loads(stdout)
            
            if response.get("status") == "error":
                error_message = response.get("data", {}).get("message", "Unknown error")
                raise RuntimeError(f"MCP server error: {error_message}")
                
            return response.get("data", {})
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise self.TimeoutError(f"MCP server {server_type}/{command} timed out after {self.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Error running MCP command {server_type}/{command}: {str(e)}")
    
    def _setup_repository(self, repo_url: str, commit_hash: str = None) -> str:
        """
        Set up a git repository for testing
        
        Args:
            repo_url: URL of the repository
            commit_hash: Git commit hash to checkout (optional)
        
        Returns:
            Path to the cloned repository
        """
        # Create a unique directory name based on the repo URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        unique_id = str(int(time.time()))
        repo_path = os.path.join(self.repos_path, f"{repo_name}_{unique_id}")
        
        if self.use_mcp:
            # Use MCP Git server to clone the repo
            self._run_mcp_command("git", "clone", {
                "repo_url": repo_url,
                "target_dir": repo_path,
                "force": True
            })
            
            # Checkout specific commit if provided
            if commit_hash:
                self._run_mcp_command("git", "checkout", {
                    "target_dir": repo_path,
                    "commit_or_branch": commit_hash
                })
        else:
            # Use subprocess to run git commands directly
            os.makedirs(repo_path, exist_ok=True)
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            
            if commit_hash:
                subprocess.run(["git", "checkout", commit_hash], cwd=repo_path, check=True)
                
        return repo_path
    
    def _apply_patch(self, repo_path: str, patch_content: str) -> bool:
        """
        Apply a patch to a repository
        
        Args:
            repo_path: Path to the repository
            patch_content: Patch content (diff format)
        
        Returns:
            True if the patch was applied successfully, False otherwise
        """
        if self.use_mcp:
            try:
                self._run_mcp_command("git", "apply_diff", {
                    "target_dir": repo_path,
                    "diff_content": patch_content
                })
                return True
            except Exception as e:
                if self.verbose:
                    print(f"Failed to apply patch: {str(e)}")
                return False
        else:
            # Write patch to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as tmp:
                tmp.write(patch_content)
                tmp_path = tmp.name
            
            try:
                # Try to apply the patch
                result = subprocess.run(
                    ["git", "apply", tmp_path],
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )
                
                success = result.returncode == 0
                if not success and self.verbose:
                    print(f"Git apply stderr: {result.stderr}")
                
                # Clean up the temporary file
                os.unlink(tmp_path)
                
                return success
            except Exception as e:
                # Clean up the temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
                if self.verbose:
                    print(f"Failed to apply patch: {str(e)}")
                
                return False
    
    def _run_test(self, repo_path: str, test_command: str) -> Dict[str, Any]:
        """
        Run tests in the repository
        
        Args:
            repo_path: Path to the repository
            test_command: Command to run the tests
        
        Returns:
            Dictionary with test results
        """
        if self.use_mcp:
            try:
                return self._run_mcp_command("process", "run_command", {
                    "command": test_command,
                    "cwd": repo_path,
                    "timeout": self.timeout,
                    "shell": True
                })
            except Exception as e:
                if self.verbose:
                    print(f"Failed to run test: {str(e)}")
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "error": str(e)
                }
        else:
            try:
                result = subprocess.run(
                    test_command,
                    cwd=repo_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=True,
                    timeout=self.timeout
                )
                
                return {
                    "returncode": result.returncode,
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip()
                }
            except subprocess.TimeoutExpired:
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Test execution timed out",
                    "timeout": True
                }
            except Exception as e:
                if self.verbose:
                    traceback.print_exc()
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "error": str(e)
                }
    
    def _cleanup_repository(self, repo_path: str):
        """Clean up repository after testing"""
        try:
            if os.path.exists(repo_path):
                shutil.rmtree(repo_path)
        except Exception as e:
            if self.verbose:
                print(f"Failed to clean up repository: {str(e)}")
    
    def extract_patch(self, text: str) -> str:
        """
        Extract patch content from agent output
        
        Args:
            text: Raw agent output text
        
        Returns:
            Extracted patch/diff content or empty string if not found
        """
        # Try to find content within diff/patch markers
        diff_pattern = r'(?:```diff|```patch)(.*?)```'
        diff_match = re.search(diff_pattern, text, re.DOTALL)
        if diff_match:
            return diff_match.group(1).strip()
        
        # Look for content that starts with diff/patch headers
        patch_pattern = r'(?:diff\s+--git\s+|---\s+\S+\s+\+\+\+\s+\S+)(.*)'
        patch_match = re.search(patch_pattern, text, re.DOTALL)
        if patch_match:
            return "diff --git " + patch_match.group(0).strip()
        
        # Extract any large code block if diff not found
        code_pattern = r'```(?:\w+)?\s*(.*?)```'
        code_match = re.search(code_pattern, text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Default to returning the full text
        return text
    
    def evaluate_patch(self, problem: Dict[str, Any], patch: str) -> Dict[str, Any]:
        """
        Evaluate a patch against a SWE-bench problem
        
        Args:
            problem: Problem definition including repository and test info
            patch: Patch/diff content to apply
        
        Returns:
            Dictionary with evaluation results
        """
        repo_url = problem.get("repo_url")
        commit_hash = problem.get("commit_hash")
        test_command = problem.get("test_command")
        
        if not repo_url or not test_command:
            return {
                "success": False,
                "error": "Missing required problem fields: repo_url and test_command"
            }
        
        repo_path = None
        try:
            # Set up repository
            repo_path = self._setup_repository(repo_url, commit_hash)
            
            # Apply the patch
            patch_success = self._apply_patch(repo_path, patch)
            if not patch_success:
                return {
                    "success": False,
                    "error": "Failed to apply patch"
                }
            
            # Run the test
            test_result = self._run_test(repo_path, test_command)
            
            # Check if test passed
            test_passed = test_result.get("returncode") == 0
            
            return {
                "success": test_passed,
                "test_result": test_result,
                "patch": patch
            }
            
        finally:
            # Clean up
            if repo_path and os.path.exists(repo_path):
                self._cleanup_repository(repo_path)
    
    def process_agent_solution(self, solution: Dict[str, Any]) -> str:
        """
        Process an agent's solution to extract the patch
        
        Handles different agent output formats and normalizes them to a standard patch format
        
        Args:
            solution: Solution from the agent system
        
        Returns:
            Extracted patch content
        """
        # Handle different agent output formats
        if isinstance(solution, dict):
            # If solution is a dictionary, check for known fields
            if "patch" in solution:
                return solution["patch"]
            elif "diff" in solution:
                return solution["diff"]
            elif "output" in solution:
                return self.extract_patch(solution["output"])
            elif "final_answer" in solution:
                return self.extract_patch(solution["final_answer"])
            elif "answer" in solution:
                return self.extract_patch(solution["answer"])
            else:
                # Convert the dict to JSON and extract patch
                return self.extract_patch(json.dumps(solution))
        elif isinstance(solution, str):
            # Direct string solution
            return self.extract_patch(solution)
        else:
            # Try to convert to string
            return self.extract_patch(str(solution))
    
    def calculate_score(self, problem: Dict[str, Any], prediction: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate score for the prediction
        
        Args:
            problem: Problem definition
            prediction: Raw prediction text
        
        Returns:
            Tuple of (score, details)
        """
        # Extract patch from prediction
        patch = self.process_agent_solution(prediction)
        
        # Evaluate the patch
        result = self.evaluate_patch(problem, patch)
        
        # Return score (1.0 if successful, 0.0 otherwise) and details
        score = 1.0 if result.get("success", False) else 0.0
        
        return score, result
    
    def create_run(self, problem: Dict[str, Any], prediction: str, patch: str, score: float, 
                  details: Dict[str, Any]) -> Run:
        """
        Create a LangSmith run for evaluation
        
        Args:
            problem: Problem definition
            prediction: Raw prediction text 
            patch: Extracted patch
            score: Evaluation score
            details: Evaluation details
        
        Returns:
            LangSmith Run object
        """
        import uuid
        
        return Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem},
            outputs={
                "prediction": prediction,
                "extracted_patch": patch,
                "score": score,
                "passed": score == 1.0,
                "details": details
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            trace_id=str(uuid.uuid4()),
        )
    
    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a problem given the agent's response
        
        Args:
            problem: Problem definition
            run_result: Agent's run result
        
        Returns:
            Evaluation results
        """
        # Extract the final answer
        prediction = run_result.get("final_answer", "")
        if not prediction and "output" in run_result:
            prediction = run_result.get("output", "")
        
        # Extract and clean patch
        patch = self.process_agent_solution(prediction)
        
        # Calculate score
        score, details = self.calculate_score(problem, prediction)
        
        # Create LangSmith run
        run = self.create_run(problem, prediction, patch, score, details)
        run_evaluation = self.run_evaluator.evaluate_run(run=run)
        
        # Return evaluation results
        return {
            "final_answer": prediction,
            "extracted_patch": patch,
            "score": score,
            "details": details,
            "run_evaluation": run_evaluation,
        } 