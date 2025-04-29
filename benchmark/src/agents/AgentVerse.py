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
            1. Generate expert descriptions in different fields based on problem requirements
            2. If feedback is provided, adapt the team composition to address the feedback
            3. Output in dict format, structure must contain agents array
            4. Each expert contains 2 fields: name, describe
            5. The description includes the specific division of labor needed to solve the problem"""
        )
        # Create a regular LLM without structured output
        self.llm = ChatOpenAI(model=self.model_name)
    def _create_prompt(self, problem: str, feedback: str = None) -> str:
        feedback_section = ""
        if feedback:
            feedback_section = f"""
            Previous evaluation feedback:
            {feedback}
            
            IMPORTANT: Consider this feedback when forming your new team of experts.
            You may need to completely change the experts or adjust their roles and responsibilities.
            """
            
        return f"""
            Generate the configuration of {self.num_agents} expert agents based on the following problem requirements:

            Problem description:
            {problem}
            
            {feedback_section}

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
            Think carefully about the problem step by step. Describe in detail the roles of different experts.
            If feedback was provided, make sure your new team addresses those specific concerns.

            Agent ID: {self.agent_id}
        """

    def describe(self, problem: str, feedback: str = None):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_prompt(problem, feedback))
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

class Evaluator:
    """Evaluates agent solutions and decides whether to recruit new experts or provide final solution"""
    def __init__(self, model_name: str = None, max_iterations: int = 3):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.max_iterations = max_iterations
        self.llm = ChatOpenAI(model=self.model_name)
        
    def evaluate(self, problem: str, solutions: List[Dict[str, Any]], iteration: int) -> Dict[str, Any]:
        """
        Evaluate solutions from multiple agents and decide whether to:
        1. Provide final solution if satisfactory
        2. Provide feedback for another round of recruitment
        
        Args:
            problem: Original problem description
            solutions: List of solutions from agents
            iteration: Current iteration count
            
        Returns:
            Dictionary with evaluation results
        """
        solutions_text = "\n\n".join([f"Expert {sol['agent_id']} solution:\n{sol['solution']}" for sol in solutions])
        
        prompt = f"""
        I need you to analyze multiple solutions to the same problem.
        
        The original problem:
        {problem}
        
        The solutions from different experts:
        {solutions_text}
        
        This is iteration {iteration} out of {self.max_iterations}.
        
        Please carefully analyze these solutions and decide:
        1. If the solutions adequately address the problem, provide a final answer
        2. If the solutions need improvement, provide feedback for recruiting better experts
        
        YOUR RESPONSE MUST BE VALID JSON following this format:
        {{
            "status": "complete" or "need_new_experts",
            "final_solution": "IMPORTANT: This should contain the COMPLETE and DETAILED solution to the original problem, not just comments about correctness. Include only the final answer as: \\\\boxed{{answer}}",
            "feedback": "Feedback for expert recruitment if status is need_new_experts, otherwise leave empty",
            "reasoning": "Your reasoning for the decision"
        }}
        
        DO NOT wrap your JSON in markdown code blocks (```). Just respond with the raw JSON object.
        Ensure your JSON is properly formatted - escape all special characters like backslashes with double backslashes.
        Be critical in your evaluation. If any important aspects of the problem remain unsolved, indicate that we need new experts.
        
        IMPORTANT: If you decide the solutions are adequate (status="complete"), your final_solution MUST include:
        1. Complete step-by-step reasoning
        2. Detailed mathematical explanation
        3. The final answer in a boxed format: \\\\boxed{{answer}}
        """
        
        messages = [
            SystemMessage(content="You are an expert evaluator that analyzes multiple solutions, determines if they're adequate, and provides feedback for improvement if needed. Always respond with valid JSON only, not wrapped in code blocks. When solutions are adequate, your final_solution must contain the complete mathematical solution with step-by-step reasoning and the final answer in a boxed format (\\boxed{answer})."),
            HumanMessage(content=prompt)
        ]
        
        start_time = time.time()
        response = self.llm.invoke(messages)
        end_time = time.time()
        
        # Set name on the AIMessage
        response.name = "evaluator"
        
        # Parse the content as JSON to get structured data
        try:
            # Clean the response by removing markdown code blocks if present
            content = response.content
            
            # Remove markdown code blocks if present (```json ... ```)
            import re
            content = re.sub(r'```(?:json)?', '', content)
            content = content.strip()
            content = re.sub(r'```$', '', content).strip()
            
            # print("Cleaned content for parsing:", content[:100] + "..." if len(content) > 100 else content)
            
            # Try direct JSON parsing first
            try:
                evaluation = json.loads(content)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON object with regex
                json_match = re.search(r'({.*})', content.replace('\n', ' '), re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                    # Handle any problematic escape sequences
                    json_str = json_str.replace('\\', '\\\\')
                    try:
                        evaluation = json.loads(json_str)
                    except:
                        # If still fails, use a more lenient approach
                        print("Using more lenient JSON parsing")
                        import ast
                        evaluation = ast.literal_eval(json_str)
                else:
                    raise ValueError("Could not find JSON object in response")
                
            # Ensure required fields exist
            if "status" not in evaluation:
                evaluation["status"] = "need_new_experts" if iteration < self.max_iterations else "complete"
            if "final_solution" not in evaluation:
                evaluation["final_solution"] = ""
            if "feedback" not in evaluation:
                evaluation["feedback"] = ""
            if "reasoning" not in evaluation:
                evaluation["reasoning"] = "No reasoning provided"
                
        except Exception as e:
            print(f"Error parsing JSON evaluation: {e}")
            print(f"Raw response: {response.content[:200]}...")
            # Default values if parsing fails
            evaluation = {
                "status": "need_new_experts" if iteration < self.max_iterations else "complete",
                "final_solution": response.content if iteration >= self.max_iterations else "",
                "feedback": f"Error parsing previous response. Please provide a team of experts that can solve this problem: {problem[:200]}..." if iteration < self.max_iterations else "",
                "reasoning": "Error parsing structured evaluation"
            }
        
        return {
            "final_solution": evaluation.get("final_solution", ""),
            "message": response,
            "latency_ms": (end_time - start_time) * 1000,
            "evaluation": evaluation,
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
        self.max_iterations = self.config.get("max_iterations", 3)
        
        # Initialize evaluator and metrics collector through base class methods
        self._initialize_evaluator()
        self._initialize_metrics_collector()
    
    def _create_agents(self, problem: str, feedback: str = None) -> Dict[str, Any]:
        """
        Create specialized work agents based on the problem and optional feedback
        
        Args:
            problem: Original problem description
            feedback: Optional feedback from previous evaluation
            
        Returns:
            Dictionary with workers and message
        """
        # Use recruiter to determine agent profiles
        recruiter = RecruiterAgent(
            agent_id="recruiter_001", 
            model_name=self.model_name,
            num_agents=self.num_agents
        )
        response_dict = recruiter.describe(problem, feedback)
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
        
        # Initialize messages and solutions
        all_messages = []
        all_solutions = []
        feedback = None
        final_solution = None
        
        # Create evaluator
        evaluator = Evaluator(model_name=self.model_name, max_iterations=self.max_iterations)
        
        # Run iterations until evaluator is satisfied or max iterations reached
        for iteration in range(1, self.max_iterations + 1):
            print(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Create specialized agents for this problem with feedback from previous iteration
            # Don't combine problem and feedback, but pass feedback separately
            recruiter_response = self._create_agents(problem_text, feedback)
            agents = recruiter_response.get("workers", [])
            recruiter_message = recruiter_response.get("message", None)
            
            # Add recruiter message to all messages
            if recruiter_message:
                all_messages.append(recruiter_message)
            
            # Run agents either in parallel or sequentially
            if self.use_parallel:
                # Set up async execution
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                # Run all agents asynchronously - don't pass feedback to workers
                agent_solutions = loop.run_until_complete(
                    self._async_solve_problem(problem_text, agents)
                )
            else:
                # Run agents sequentially - don't pass feedback to workers
                agent_solutions = []
                for agent in agents:
                    solution = agent.solve(problem_text)
                    agent_solutions.append(solution)
            
            # Collect agent messages and solutions for this iteration
            iteration_messages = []
            for solution in agent_solutions:
                if "message" in solution:
                    iteration_messages.append(solution["message"])
                    all_messages.append(solution["message"])
            
            # Store solutions for current iteration
            all_solutions.append({
                "iteration": iteration,
                "solutions": agent_solutions
            })
            
            # Evaluate solutions
            evaluation_result = evaluator.evaluate(problem_text, agent_solutions, iteration)
            evaluation = evaluation_result.get("evaluation", {})
            
            # Add evaluator message
            if "message" in evaluation_result:
                all_messages.append(evaluation_result["message"])
            
            # Check if we need another iteration
            status = evaluation.get("status", "need_new_experts")
            
            if status == "complete":
                final_solution = evaluation.get("final_solution", "")
                print(f"Evaluation complete after {iteration} iterations")
                break
            else:
                feedback = evaluation.get("feedback", "")
        
        # If we reached max iterations without a satisfactory solution, use the last evaluation
        if final_solution is None and all_solutions:
            last_evaluation = evaluator.evaluate(problem_text, all_solutions[-1]["solutions"], self.max_iterations)
            final_solution = last_evaluation.get("evaluation", {}).get("final_solution", "No satisfactory solution found")
            # Add final evaluator message
            if "message" in last_evaluation:
                all_messages.append(last_evaluation["message"])
        
        # For math problems, ensure the final solution is properly formatted
        if isinstance(final_solution, (int, float)):
            final_solution = f"The answer is \\boxed{{{final_solution}}}"
            
        # Return final answer and all messages
        return {
            "messages": all_messages,
            "final_answer": final_solution,
            "agent_solutions": all_solutions,
        }

# Register the agent system
AgentSystemRegistry.register("agentverse", AgentVerse, num_agents=3, parallel=True, max_iterations=100)








