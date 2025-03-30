"""
Supervisor-based Multi-Agent System

This module implements a supervisor-based multi-agent system where a supervisor
agent coordinates the work of specialized agents.
"""

import time
import uuid
import os
from typing import Literal, Dict, TypedDict, Any
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable, RunTree
from langsmith.evaluation import RunEvaluator
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.schemas import Run
from dotenv import load_dotenv
import asyncio
from benchmark.src.evaluators.math import MATHBenchmark
import dotenv
dotenv.load_dotenv()

# Import the base agent system classes
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

# Optional import for metrics instrumentation
try:
    from benchmark.src.metrics import (
        MetricsRegistry,
        AgentMetricsCollector,
        InterAgentMetricsCollector
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# root_path = Path(__file__).parent

# project_root = Path(__file__).parent.parent

# sys.path.append(str(root_path))


load_dotenv()

# tavily_tool = TavilySearchResults(max_results=5)


class State(MessagesState):
    next: str
    token_usage: Dict[str, int]


class Router(TypedDict):
    next: Literal["researcher", "coder", "FINISH"]


# Metrics instrumentation decorator
def with_metrics(func):
    def wrapper(*args, **kwargs):
        if not METRICS_AVAILABLE:
            return func(*args, **kwargs)
        
        metrics_registry = kwargs.get('metrics_registry')
        if not metrics_registry:
            return func(*args, **kwargs)
        
        agent_collector = metrics_registry.get_collector("agent")
        inter_agent_collector = metrics_registry.get_collector("inter_agent")
        
        # Record function start
        start_time = time.time()
        
        # Run the original function
        result = func(*args, **kwargs)
        
        # Record function completion
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        if 'token_usage' in result:
            for agent_id, tokens in result['token_usage'].items():
                agent_collector.record_llm_usage(
                    agent_id=agent_id,
                    model_name="gpt-4o-mini",
                    prompt_tokens=tokens // 2,
                    completion_tokens=tokens // 2,
                    latency_ms=duration_ms / len(result['token_usage']),
                )
        
        return result
    
    return wrapper


@traceable
def create_supervisor():
    model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), 
                      base_url=os.getenv("BASE_URL"), 
                      api_key=os.getenv("OPENAI_API_KEY"))
    members = ["researcher", "coder"]

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH.\n\n"
        "For mathematical problems, you should first send the request to the 'coder' "
        "who can solve mathematical problems and provide formatted answers. "
        "Only use the 'researcher' if additional information needs to be looked up. "
        "The 'coder' is a mathematical expert who can solve problems directly."
    )

    def supervisor_node(state: State) -> Command[Literal["researcher", "coder", "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        start_time = time.time()
        with get_openai_callback() as cb:
            response = model.with_structured_output(Router).invoke(messages)
        duration_ms = (time.time() - start_time) * 1000

        # Record interaction for metrics collection (if available)
        if METRICS_AVAILABLE:
            # This is a placeholder for metrics collection
            # In a real implementation, you'd access the metrics registry here
            pass

        state["token_usage"]["supervisor"] += cb.total_tokens
        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto, "token_usage": state["token_usage"]})

    return supervisor_node


@traceable
def create_research_node():
    model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), 
                      base_url=os.getenv("BASE_URL"), 
                      api_key=os.getenv("OPENAI_API_KEY"))
    research_agent = create_react_agent(model, tools=[])

    def research_node(state: State) -> Command[Literal["supervisor"]]:
        start_time = time.time()
        with get_openai_callback() as cb:
            result = research_agent.invoke(state)
        duration_ms = (time.time() - start_time) * 1000

        state["token_usage"]["researcher"] += cb.total_tokens

        return Command(
            update={
                "messages": [HumanMessage(content=result["messages"][-1].content, name="researcher")],
                "token_usage": state["token_usage"],
            },
            goto="supervisor",
        )

    return research_node


@traceable
def create_code_node():
    model = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), 
                      base_url=os.getenv("BASE_URL"), 
                      api_key=os.getenv("OPENAI_API_KEY"))

    coder_agent = create_react_agent(
        model,
        tools=[],
    )

    def code_node(state: State) -> Command[Literal["supervisor"]]:
        start_time = time.time()
        with get_openai_callback() as cb:
            result = coder_agent.invoke(state)
        duration_ms = (time.time() - start_time) * 1000

        state["token_usage"]["coder"] += cb.total_tokens

        return Command(
            update={
                "messages": [HumanMessage(content=result["messages"][-1].content, name="coder")],
                "token_usage": state["token_usage"],
            },
            goto="supervisor",
        )

    return code_node


def create_mas_graph():
    """Create the multi-agent system graph"""
    builder = StateGraph(State)
    checkpointer = InMemorySaver()

    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", create_supervisor())
    builder.add_node("researcher", create_research_node())
    builder.add_node("coder", create_code_node())

    return builder.compile(checkpointer=checkpointer)


class SupervisorMAS(AgentSystem):
    """
    Supervisor-based Multi-Agent System
    
    This agent system uses a supervisor to coordinate specialized agents
    for solving problems.
    """
    
    def __init__(self, name: str = "supervisor_mas", config: Dict[str, Any] = None):
        """Initialize the Supervisor MAS"""
        super().__init__(name, config)
        self.evaluator_name = config.get("evaluator", "math") if config else "math"
    
    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate the agent system on a problem"""
        metrics_registry = kwargs.get("metrics_registry", self.metrics_registry)
        
        # Initialize evaluator
        run_evaluator = RunEvaluator()
        math_evaluator = MATHBenchmark(
            name=self.evaluator_name.upper(),
            file_path=f"benchmark/src/data/{self.evaluator_name}_test.jsonl",
            log_path=f"benchmark/src/data/results/{self.evaluator_name.upper()}",
        )

        # Create graph
        graph = create_mas_graph()
        initial_state = {
            "messages": [("user", problem["problem"])],
            "token_usage": {"supervisor": 0, "researcher": 0, "coder": 0},
        }

        # Record start time for metrics
        start_time = time.time()
        
        # Run the graph
        run_result = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
        
        # Record execution time
        execution_time_ms = self.record_timing("evaluate", start_time, 
                                            {"problem_id": problem.get("id", "unknown")})

        # Extract the final answer
        all_messages = run_result.get("messages", [])
        final_answer = ""

        if all_messages:
            last_msg = all_messages[-1]
            if isinstance(last_msg, tuple) and len(last_msg) > 1:
                final_answer = last_msg[1]
            elif hasattr(last_msg, "content"):
                final_answer = last_msg.content
            elif isinstance(last_msg, dict) and "content" in last_msg:
                final_answer = last_msg["content"]

        # Calculate score
        score, extracted_answer = math_evaluator.calculate_score(problem["solution"], final_answer)

        # Extract token usage
        token_usage = run_result.get("token_usage", {})
        
        # Record token usage in metrics
        if metrics_registry and METRICS_AVAILABLE:
            agent_collector = metrics_registry.get_collector("agent")
            if agent_collector:
                for agent_id, tokens in token_usage.items():
                    agent_collector.record_llm_usage(
                        agent_id=agent_id,
                        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                        prompt_tokens=tokens,
                        completion_tokens=0,
                        latency_ms=execution_time_ms / len(token_usage) if token_usage else 0,
                        tags={"agent_system": self.name}
                    )
        
        # Create LangSmith run for evaluation
        run = Run(
            id=self.generate_run_id(),
            name=f"{self.evaluator_name.upper()}_MAS_Evaluation",
            inputs={"problem": problem["problem"]},
            outputs={
                "prediction": final_answer,
                "extracted_answer": extracted_answer,
                "expected": problem["solution"],
                "math_score": score,
                "passed": score == 1,
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
            "execution_time_ms": execution_time_ms
        }


# Register the agent system
AgentSystemRegistry.register(
    "supervisor_mas", 
    SupervisorMAS,
    evaluator="math"
)


# Legacy function for backward compatibility
def evaluate_mas(problem: dict, metrics_registry=None):
    """Legacy function for backward compatibility"""
    agent = SupervisorMAS()
    if metrics_registry:
        agent.set_metrics_registry(metrics_registry)
    return agent.evaluate(problem, metrics_registry=metrics_registry)


if __name__ == "__main__":
    import json

    # Test the agent system
    with open("benchmark/src/data/math_test.jsonl", "r") as f:
        problems = [json.loads(line) for line in f]

    # Process only a few problems for testing
    all_results = []
    test_problems = problems[:3]  # Just process 3 problems for testing

    for i, problem in enumerate(test_problems):
        print(f"\nProcessing Problem {i + 1}/{len(test_problems)}")
        print(f"Problem: {problem['problem'][:100]}...")

        try:
            # Create agent and run evaluation
            agent = SupervisorMAS()
            results = agent.evaluate(problem)

            # Save results
            all_results.append({
                "problem": problem["problem"],
                "expected": problem["solution"],
                "prediction": results["extracted_answer"],
                "passed": results["math_score"] == 1,
                "evaluation": results["run_evaluation"],
                "token_usage": results["token_usage"],
                "execution_time_ms": results["execution_time_ms"]
            })

            print(f"Expected: {problem['solution']}")
            print(f"Predicted: {results['extracted_answer']}")
            print(f"Passed: {results['math_score'] == 1}")
            print(f"Execution time: {results['execution_time_ms']:.2f}ms")

        except Exception as e:
            print(f"Error processing problem {i + 1}: {str(e)}")
            continue

    # Print statistics
    total = len(test_problems)
    passed = sum(1 for r in all_results if r["passed"])
    total_tokens = sum(sum(d.values()) for r in all_results if "token_usage" in r for d in [r["token_usage"]])
    total_time = sum(r.get("execution_time_ms", 0) for r in all_results)
    
    print(f"\nSummary:")
    print(f"Passed: {passed}/{total} ({passed / total:.2%})")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per problem: {total_tokens / total:.2f}")
    print(f"Total execution time: {total_time:.2f}ms")
    print(f"Average execution time: {total_time / total:.2f}ms per problem")
