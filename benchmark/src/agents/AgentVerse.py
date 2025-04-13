import time
import json
import os
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, TypedDict, Any, List
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Annotated, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.manager import AsyncCallbackManager
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()

@dataclass
class ExpertProfile:
    id: str
    name: str
    description: str

class Agent(BaseModel):
    name: str
    describe: str
    agent_id: int

class Agents(BaseModel):
    agents: List[Agent]

class Discussion(TypedDict):
    agent_id: int
    context: str

class SumDiscussion(TypedDict):
    sum_context: List[Discussion]

class RecruiterAgent:
    """Recruitment agent: generates descriptions for work agents"""
    def __init__(self, agent_id: str, model_name: str = None, num_agents: int = 3):
        self.agent_id = agent_id
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.num_agents = num_agents
        self.system_prompt = (
            "You are a professional AI recruitment expert who needs to generate the right work team configuration based on the needs of the problem."
            """Please strictly follow the following rules:
            1.Generate expert descriptions in different fields at a time,
            2.Output in dict format, structure must contain agents array
            3.Each expert contains 2 fields: name, describe
            4.The description includes the specific division of labor needed to solve the problem"""
        )
        # Create a regular LLM without structured output
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def _create_prompt(self, problem: str) -> str:
        return f"""
            Generate the configuration of {self.num_agents} expert agents based on the following problem requirements:

            Problem description:
            {problem}

            Please respond in the following dict format:

            {{
  "agents": [
    {{
      "name": "Expert name",
      "describe": "Expert description",
      "agent_id":"start from 1(eg: 1)"
    }},
    // Residual same structure
  ]
}}
            Think carefully about the problem step by step. Describe in detail the roles of different experts

            Agent ID: {self.agent_id}
        """

    def describe(self, problem: str):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_prompt(problem))
        ]

        start_time = time.time()
        # Get the raw AIMessage response which contains metadata
        response = self.llm.invoke(messages)
        end_time = time.time()

        
        # Set name on the AIMessage
        response.name = f"recruiter_{self.agent_id}"
        
        # Parse the content as JSON to get structured data
        try:
            content_json = json.loads(response.content)
            structured_data = {"agents": []}
            
            # Process the agents data
            if "agents" in content_json:
                structured_data["agents"] = content_json["agents"]
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            # If JSON parsing fails, create a fallback structure
            structured_data = {"agents": []}
        return {
            "agent_id": self.agent_id,
            "solution": structured_data,
            "message": response,
            "latency_ms": (end_time - start_time) * 1000,
        }

class WorkAgent:
    """Work agent that solves specific aspects of a problem"""
    def __init__(self, agent_id: str, system_prompt: str = None):
        self.agent_id = agent_id
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            f"{system_prompt}\n"
            "## Output Requirements:\n"
            "1. Solve the appropriate part of the problem according to your division of labor within 800 tokens\n"
            "2. Use clear section headers\n"
            "3. Prioritize key conclusions first"
        )
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=1000
        )

    def solve(self, problem: str, feedback: str = None):
        """Solve a problem with optional feedback"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=problem + (feedback or ""))
        ]

        start_time = time.time()
        response = self.llm.invoke(messages)
        end_time = time.time()

        
        # Create a proper AIMessage for consistent interface with other agents
        ai_message = response 
        ai_message.name = f"expert_{self.agent_id}"
        
        return {
            "agent_id": self.agent_id,
            "solution": response.content,
            "message": ai_message,
            "latency_ms": (end_time - start_time) * 1000,
        }

class Aggregator:
    """Aggregates results from work agents to produce a final solution"""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def aggregate(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate solutions from multiple agents"""
        solutions_text = "\n\n".join([f"Expert {sol['agent_id']} solution:\n{sol['solution']}" for sol in solutions])

        prompt = f"""
        I need you to analyze multiple solutions to the same problem and provide the most accurate answer.

        The original problem:
        {problem}

        The solutions from different experts:
        {solutions_text}

        Please carefully analyze these solutions, identify the correct approach, and provide the final answer.
        Make sure your final answer is clearly formatted and precise. Only include the final answer.
        """

        messages = [
            {
                "role": "system",
                "content": "You are an expert aggregator that analyzes multiple solutions and determines the most accurate one.",
            },
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        response = self.llm.invoke(messages)
        end_time = time.time()

        
        # Create a proper AIMessage for the aggregator
        ai_message = response
        ai_message.name = "aggregator"

        return {
            "final_solution": response.content,
            "message": ai_message,
            "latency_ms": (end_time - start_time) * 1000,
        }

class AgentVerse(AgentSystem):
    """
    AgentVerse Multi-Agent System
    
    This agent system uses a recruiter to create specialized agents for different aspects 
    of a problem, with results aggregated to produce a final solution.
    """
    
    def __init__(self, name: str = "agentverse", config: Dict[str, Any] = None):
        """Initialize the AgentVerse System"""
        super().__init__(name, config)
        self.config = config or {}
        self.evaluator_name = self.config.get("evaluator", "math")
        self.num_agents = self.config.get("num_agents", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.use_parallel = self.config.get("parallel", True)
        
        # Initialize evaluator and metrics collector through base class methods
        self._initialize_evaluator()
        self._initialize_metrics_collector()

    def _create_agents(self, problem: str) -> List[WorkAgent]:
        """Create specialized work agents based on the problem"""
        # Use recruiter to determine agent profiles
        recruiter = RecruiterAgent(
            agent_id="recruiter_001", 
            model_name=self.model_name,
            num_agents=self.num_agents
        )
        response_dict = recruiter.describe(problem)
        agents_list = response_dict.get("solution", {}).get("agents", [])
        expert_team = [
            ExpertProfile(
                id=str(agent.get("agent_id", "000")),
                name=agent.get("name", "Unnamed Expert").strip(),
                description=agent.get("describe", "")[:500]  # Truncate long descriptions
            ) for agent in agents_list
            if isinstance(agent, dict)
        ]
        
        # Create work agents based on profiles
        workers = []
        for expert in expert_team:
            workers.append(
                WorkAgent(
                    agent_id=expert.id,
                    system_prompt=expert.description
                )
            )
        return {"workers": workers, "message": response_dict.get("message", None)}

    async def _solve_async(self, worker: WorkAgent, problem: str, feedback: str = None) -> Dict[str, Any]:
        """Solve a problem asynchronously with a worker agent"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, worker.solve, problem, feedback)

    async def _async_solve_problem(self, problem: str, workers: List[WorkAgent], feedback: str = None) -> List[Dict[str, Any]]:
        """Solve a problem with multiple worker agents asynchronously"""
        # Create tasks for each worker
        tasks = [asyncio.create_task(self._solve_async(worker, problem, feedback)) for worker in workers]
        
        # Run all tasks concurrently
        solutions = await asyncio.gather(*tasks)
        
        return solutions

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
        
        # Create specialized agents for this problem
        recruiter_response = self._create_agents(problem_text)
        agents = recruiter_response.get("workers", [])
        recruiter_message = recruiter_response.get("message", None)
        
        # Collect agent solutions and messages
        all_messages = []
        all_messages.append(recruiter_message)
        
        # Run agents either in parallel or sequentially
        if self.use_parallel:
            # Set up async execution
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Run all agents asynchronously
            agent_solutions = loop.run_until_complete(
                self._async_solve_problem(problem_text, agents)
            )
        else:
            # Run agents sequentially
            agent_solutions = []
            for agent in agents:
                solution = agent.solve(problem_text)
                agent_solutions.append(solution)
        
        # Collect all agent messages
        for solution in agent_solutions:
            if "message" in solution:
                all_messages.append(solution["message"])
        
        # Aggregate solutions
        aggregator = Aggregator(model_name=self.model_name)
        aggregated = aggregator.aggregate(problem_text, agent_solutions)
        
        # Add aggregator message
        if "message" in aggregated:
            all_messages.append(aggregated["message"])
        
        # Return final answer and all messages
        return {
            "messages": all_messages,
            "final_answer": aggregated["final_solution"],
            "agent_solutions": [{"agent_id": s["agent_id"], "solution": s["solution"]} for s in agent_solutions],
        }

# Register the agent system
AgentSystemRegistry.register("agentverse", AgentVerse, num_agents=3, parallel=True)








