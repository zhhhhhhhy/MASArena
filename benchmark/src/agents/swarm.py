"""
Swarm Agent System

This module implements a swarm-based multi-agent system where multiple agents
work collaboratively to solve problems, with each agent working independently
and then aggregating their results.
"""

import time
import uuid
import os
import json
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from dotenv import load_dotenv

from benchmark.src.evaluators.math import MATHBenchmark
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SwarmAgent:
    """Individual agent in the swarm"""

    def __init__(self, agent_id: str, model_name: str = None, system_prompt: str = None):
        """
        Initialize a swarm agent.

        Args:
            agent_id: Unique identifier for this agent
            model_name: LLM model to use
            system_prompt: Custom system prompt for this agent
        """
        self.agent_id = agent_id
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            system_prompt
            or "You are an intelligent AI assistant specialized in solving problems carefully and step by step."
        )
        self.llm = ChatOpenAI(
            model=self.model_name, base_url=os.getenv("BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        )
        self.token_usage = 0

    def solve(self, problem: str) -> Dict[str, Any]:
        """
        Solve a problem independently.

        Args:
            problem: The problem to solve

        Returns:
            Dictionary with the solution and metadata
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._create_prompt(problem)},
        ]

        start_time = time.time()
        with get_openai_callback() as cb:
            response = self.llm.invoke(messages)
        end_time = time.time()

        self.token_usage = cb.total_tokens

        return {
            "agent_id": self.agent_id,
            "solution": response.content,
            "token_usage": self.token_usage,
            "latency_ms": (end_time - start_time) * 1000,
        }

    def _create_prompt(self, problem: str) -> str:
        """Create a tailored prompt for this agent"""
        return f"""
Please solve the following problem:

{problem}

Think carefully about the problem step by step. Show your work and reasoning.
For mathematical problems, make sure to provide your final answer in a clear format.

Agent ID: {self.agent_id}
"""


class Aggregator:
    """Aggregates results from swarm agents to produce a final solution"""

    def __init__(self, model_name: str = None):
        """
        Initialize the aggregator.

        Args:
            model_name: LLM model to use for aggregation
        """
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=self.model_name, base_url=os.getenv("BASE_URL"), api_key=os.getenv("OPENAI_API_KEY")
        )
        self.token_usage = 0

    def aggregate(self, problem: str, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate solutions from multiple agents.

        Args:
            problem: Original problem
            solutions: List of agent solutions

        Returns:
            Aggregated solution and metadata
        """
        solutions_text = "\n\n".join([f"Agent {sol['agent_id']} solution:\n{sol['solution']}" for sol in solutions])

        prompt = f"""
I need you to analyze multiple solutions to the same problem and provide the most accurate answer.

The original problem:
{problem}

The solutions from different agents:
{solutions_text}

Please carefully analyze these solutions, identify the correct approach, and provide the final answer.
Make sure your final answer is clearly formatted and precise.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert aggregator that analyzes multiple solutions and determines the most accurate one.",
            },
            {"role": "user", "content": prompt},
        ]

        start_time = time.time()
        with get_openai_callback() as cb:
            response = self.llm.invoke(messages)
        end_time = time.time()

        self.token_usage = cb.total_tokens

        return {
            "final_solution": response.content,
            "token_usage": self.token_usage,
            "latency_ms": (end_time - start_time) * 1000,
        }


class SwarmSystem(AgentSystem):
    """
    Swarm Agent System

    This agent system uses multiple independent agents working in parallel,
    with results aggregated to produce a final solution.
    """

    def __init__(self, name: str = "swarm", config: Dict[str, Any] = None):
        """Initialize the Swarm Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        self.evaluator_name = self.config.get("evaluator", "math")
        self.num_agents = self.config.get("num_agents", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.use_parallel = self.config.get("parallel", True)

    def _create_agents(self) -> List[SwarmAgent]:
        """Create the swarm agents"""
        return [
            SwarmAgent(agent_id=f"agent_{i + 1}", model_name=self.model_name, system_prompt=self._get_system_prompt(i))
            for i in range(self.num_agents)
        ]

    def _get_system_prompt(self, agent_index: int) -> str:
        """Get system prompt for an agent based on its index"""
        base_prompt = "You are an intelligent AI assistant specialized in solving problems carefully and step by step."

        # Different specializations for different agents
        specializations = [
            " Focus on being methodical and detailed in your approach.",
            " Focus on finding elegant and efficient solutions.",
            " Focus on checking edge cases and validating your answers.",
            " Focus on breaking down complex problems into simpler parts.",
            " Focus on applying mathematical principles rigorously.",
        ]

        # Use the agent index to select a specialization, cycling through the options
        specialization_index = agent_index % len(specializations)
        return base_prompt + specializations[specialization_index]

    async def _solve_problem_async(self, agent: SwarmAgent, problem: str) -> Dict[str, Any]:
        """Solve a problem asynchronously"""
        # Run in a thread pool to avoid blocking
        with ThreadPoolExecutor() as executor:
            return await asyncio.get_event_loop().run_in_executor(executor, agent.solve, problem)

    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate the agent system on a problem"""
        metrics_registry = kwargs.get("metrics_registry", self.metrics_registry)
        problem_text = problem["problem"]

        # Initialize evaluator
        run_evaluator = RunEvaluator()
        math_evaluator = MATHBenchmark(
            name=self.evaluator_name.upper(),
            file_path=f"benchmark/data/{self.evaluator_name}_test.jsonl",
            log_path=f"benchmark/data/results/{self.evaluator_name.upper()}",
        )

        # Record start time
        start_time = time.time()

        # Create swarm agents
        agents = self._create_agents()

        # Collect agent solutions
        agent_solutions = []
        token_usage = {}

        if self.use_parallel:
            # Solve problems in parallel
            async def solve_all():
                tasks = [self._solve_problem_async(agent, problem_text) for agent in agents]
                return await asyncio.gather(*tasks)

            # Run the async tasks
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            agent_solutions = loop.run_until_complete(solve_all())
        else:
            # Solve problems sequentially
            for agent in agents:
                solution = agent.solve(problem_text)
                agent_solutions.append(solution)

        # Record token usage for each agent
        for solution in agent_solutions:
            token_usage[solution["agent_id"]] = solution["token_usage"]

        # Aggregate solutions
        aggregator = Aggregator(model_name=self.model_name)
        aggregated = aggregator.aggregate(problem_text, agent_solutions)
        token_usage["aggregator"] = aggregated["token_usage"]

        # Final answer
        final_answer = aggregated["final_solution"]

        # Calculate score
        score, extracted_answer = math_evaluator.calculate_score(problem["solution"], final_answer)

        # Record execution time
        execution_time_ms = self.record_timing("evaluate", start_time, {"problem_id": problem.get("id", "unknown")})

        # Record token usage in metrics
        if metrics_registry:
            agent_collector = metrics_registry.get_collector("agent")
            if agent_collector:
                for agent_id, tokens in token_usage.items():
                    agent_collector.record_llm_usage(
                        agent_id=agent_id,
                        model_name=self.model_name,
                        prompt_tokens=tokens,
                        completion_tokens=0,
                        latency_ms=execution_time_ms / len(token_usage) if token_usage else 0,
                        tags={"agent_system": self.name},
                    )

        # Create run for evaluation
        run = Run(
            id=self.generate_run_id(),
            name=f"{self.evaluator_name.upper()}_SWARM_Evaluation",
            inputs={"problem": problem_text},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["solution"],
                "math_score": score,
                "passed": score == 1,
                "agent_solutions": [s["solution"] for s in agent_solutions],
            },
            run_type="evaluation",
            start_time="2025-03-11T12:00:00Z",  # Example time, not actual
            trace_id=self.generate_run_id(),
        )

        run_evaluation = run_evaluator.evaluate_run(run=run)

        # Return evaluation results
        return {
            "final_answer": final_answer,
            "math_score": score,
            "run_evaluation": run_evaluation,
            "extracted_answer": extracted_answer,
            "token_usage": token_usage,
            "execution_time_ms": execution_time_ms,
            "agent_solutions": [{"agent_id": s["agent_id"], "solution": s["solution"]} for s in agent_solutions],
        }


# Register the agent system
AgentSystemRegistry.register("swarm", SwarmSystem, evaluator="math", num_agents=3, parallel=True)


if __name__ == "__main__":
    # Test the agent system
    with open("benchmark/data/math_test.jsonl", "r") as f:
        problems = [json.loads(line) for line in f]

    # Process only a single problem for testing
    test_problem = problems[0]
    print(f"Problem: {test_problem['problem'][:100]}...")

    # Create and run the swarm
    swarm = SwarmSystem(config={"num_agents": 2})
    results = swarm.evaluate(test_problem)

    # Print results
    print("\nAgent Solutions:")
    for agent_sol in results["agent_solutions"]:
        print(f"\nAgent {agent_sol['agent_id']}:")
        print(f"{agent_sol['solution'][:200]}...\n")

    print("\nAggregated Solution:")
    print(f"{results['final_answer'][:200]}...")

    print(f"\nExpected: {test_problem['solution']}")
    print(f"Extracted Answer: {results['extracted_answer']}")
    print(f"Score: {results['math_score']}")
    print(f"Execution time: {results['execution_time_ms']:.2f}ms")

    # Print token usage
    print("\nToken Usage:")
    for agent_id, tokens in results["token_usage"].items():
        print(f"{agent_id}: {tokens} tokens")

    total_tokens = sum(results["token_usage"].values())
    print(f"Total tokens: {total_tokens}")
