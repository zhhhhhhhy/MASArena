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
        self.llm = ChatOpenAI(model=self.model_name)

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.
        
        This method implements the actual agent logic without handling evaluation or metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
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
            {"role": "user", "content":self.format_prompt(benchmark=self.evaluator_name)},
            {"role": "user", "content": f"Problem: {problem_text}"},
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