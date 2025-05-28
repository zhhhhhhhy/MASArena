"""
Single Agent System

This module implements a simple single-agent system that uses a single LLM
to solve problems directly.
"""

import time
import uuid
import os
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SingleAgent(AgentSystem):
    """
    Single Agent System

    This agent system uses a single LLM to solve problems directly.
    """

    def __init__(self, name: str = "single_agent", config: Dict[str, Any] = None):
        """Initialize the Single Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        self.evaluator_name = self.config.get("evaluator", "bbh")  # Default to bbh
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")  # Use qwen-plus
        self.system_prompt = (
            self.config.get("system_prompt")
            or "You are an intelligent AI assistant specialized in solving complex problems step by step."
        )
        
        # Initialize evaluator and metrics collector through base class methods
        self._initialize_evaluator()
        self._initialize_metrics_collector()
        self.llm = ChatOpenAI(
           model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )
   # _create_prompt for BBH problems
    def _create_prompt(self, problem: str) -> str:
        """
        Create a structured prompt for BBH problems with clear delimiters for reasoning and answer.
        
        The model is instructed to:
        - Reason step by step within <think>...</think> tags
        - Provide the final answer only within <answer>...</answer> tags
        - Avoid any output outside these tags
        - Match the exact answer format required by the problem
        """
        return f"""
You are a highly capable AI assistant tasked with solving a problem from the Big-Bench Hard (BBH) dataset. Your response must be precise, structured, and adhere to the expected answer format for the problem type.

Instructions:
1. Analyze the problem carefully and provide your step-by-step reasoning within <think>...</think> tags.
2. Provide only the final answer within <answer>...</answer> tags, ensuring it matches the exact format required by the problem (e.g., 'True', 'False', '(A)', '(B)', a space-separated string, or a sequence of characters).
3. Do NOT include any explanation, justification, or text outside the <think> and <answer> tags.
4. Ensure the final answer is a single line with no extra whitespace or formatting.
5. Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like '> ) }}'
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'

Problem:
{problem}

Your response must follow this format:
<think>
[Your step-by-step reasoning here]
</think>
<answer>
[Your final answer here]
</answer>
"""
    # _create_prompt for math problems
    # def _create_prompt(self, problem: str) -> str:
    #         """Create a prompt for the agent"""
    #         return f"""
    # Please solve the following problem carefully and step by step:

    # {problem}

    # For mathematical problems, make sure to:
    # 1. Break down the problem into simpler parts
    # 2. Solve each part methodically
    # 3. Check your work and verify your answer
    # 4. Provide your final answer in a clear format
    # """

    def run_agent(self, problem: Dict[str, Any], problem_type: str, **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method implements the actual agent logic without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            problem_type: Type of problem (e.g., 'bbh')
            
        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]
        problem_id = problem.get("id", f"problem_{hash(problem_text)}")
        
        # Initialize the language model
        llm = self.llm

        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._create_prompt(problem_text)},
        ]

        # Get solution from LLM and track usage
        response = llm.invoke(messages)
        
        # Clean response content
        response_content = response.content.replace('\r\n', '\n').replace('\r', '\n').strip()
        try:
            response_content = response_content.encode('utf-8').decode('utf-8-sig')  # Remove BOM
        except UnicodeDecodeError:
            pass  # Ignore if already clean
        
        # print("模型返回结果:", response)  # Keep original print
        # print(f"[Debug] Cleaned response content: {repr(response_content)}")  # Debugging
        
        ai_message = response
        ai_message.name = "single_agent"
        
        # Return the response and message with usage metadata for the evaluate method
        return {
            "messages": [ai_message],
            "final_answer": response_content  # Use cleaned content
        }


# Register the agent system
AgentSystemRegistry.register("single_agent", SingleAgent)