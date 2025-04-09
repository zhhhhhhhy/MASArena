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
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langsmith import traceable, RunTree
from langsmith.evaluation import RunEvaluator
from langgraph.checkpoint.memory import InMemorySaver
from langsmith.schemas import Run
from dotenv import load_dotenv
import dotenv
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry
from benchmark.src.evaluators.math_evaluator import MathEvaluator
from benchmark.src.metrics.collector import MetricsCollector

dotenv.load_dotenv()

# Optional import for metrics instrumentation
try:
    from benchmark.src.metrics import MetricsRegistry, AgentMetricsCollector, InterAgentMetricsCollector

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

        metrics_registry = kwargs.get("metrics_registry")
        if not metrics_registry:
            return func(*args, **kwargs)

        agent_collector = metrics_registry.get_collector("agent")

        # Record function start
        start_time = time.time()

        # Run the original function
        result = func(*args, **kwargs)

        # Record function completion
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        if "token_usage" in result:
            for agent_id, tokens in result["token_usage"].items():
                agent_collector.record_llm_usage(
                    agent_id=agent_id,
                    model_name="gpt-4o-mini",
                    prompt_tokens=tokens // 2,
                    completion_tokens=tokens // 2,
                    latency_ms=duration_ms / len(result["token_usage"]),
                )

        return result

    return wrapper


@traceable
def create_supervisor():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
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

        with get_openai_callback() as cb:
            response = model.with_structured_output(Router).invoke(messages)

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
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    research_agent = create_react_agent(model, tools=[])

    def research_node(state: State) -> Command[Literal["supervisor"]]:
        with get_openai_callback() as cb:
            result = research_agent.invoke(state)

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
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    coder_agent = create_react_agent(
        model,
        tools=[],
    )

    def code_node(state: State) -> Command[Literal["supervisor"]]:
        with get_openai_callback() as cb:
            result = coder_agent.invoke(state)

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
        print("config", config)
        self.evaluator_name = config.get("evaluator", "math") if config else "math"
        
        # Initialize the appropriate evaluator
        if self.evaluator_name == "math":
            self.evaluator = MathEvaluator(
                name=self.evaluator_name,
                config={
                    "data_path": config.get("data_path", f"benchmark/data/{self.evaluator_name}_test.jsonl"),
                    "log_path": config.get("log_path", f"benchmark/data/results/{self.evaluator_name.upper()}")
                }
            )
        else:
            # Default to math evaluator for now
            # In future, we would add more evaluator types here
            self.evaluator = MathEvaluator(name=self.evaluator_name)
            
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector()

    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate the agent system on a problem"""
        metrics_registry = kwargs.get("metrics_registry", self.metrics_registry)
        
        # Set metrics registry for collector
        self.metrics_collector.set_metrics_registry(metrics_registry)

        # Create graph
        graph = create_mas_graph()
        initial_state = {
            "messages": [("user", problem["problem"])],
            "token_usage": {"supervisor": 0, "researcher": 0, "coder": 0},
        }

        # Record start time for metrics
        problem_id = problem.get("id", str(uuid.uuid4()))
        self.metrics_collector.start_timer(
            "problem_evaluation", 
            {"problem_id": problem_id, "agent_system": self.name}
        )

        # Run the graph
        run_result = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})

        # Record execution time
        execution_time_ms = self.metrics_collector.stop_timer("problem_evaluation")

        # Extract token usage
        print("run_result", run_result)
        token_usage = run_result.get("token_usage", {})
        
        for message in run_result.get("messages", []):
            if isinstance(message, AIMessage):
                # Extract token usage metadata from AIMessage
                if hasattr(message, 'usage_metadata'):
                    usage_metadata = message.usage_metadata
                    if usage_metadata:
                        print("usage_metadata", usage_metadata)
                        input_tokens = usage_metadata.get('input_tokens', 0)
                        output_tokens = usage_metadata.get('output_tokens', 0)
                        total_tokens = usage_metadata.get('total_tokens', 0)
                        
                        # Update token usage with detailed breakdown
                        token_usage[message.id] = {
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': total_tokens,
                            'input_token_details': usage_metadata.get('input_token_details', {}),
                            'output_token_details': usage_metadata.get('output_token_details', {})
                        }
                        
                        # Record token usage with centralized metrics collector
                        self.metrics_collector.record_llm_usage(
                            agent_id=message.id,
                            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=total_tokens,
                            latency_ms=execution_time_ms / len(run_result.get("messages", [])) if run_result.get("messages", []) else 0,
                            tags={"agent_system": self.name, "problem_id": problem_id}
                        )

        # Use the evaluator to evaluate the results
        evaluation_results = self.evaluator.evaluate(problem, run_result)
        
        # Record evaluation metrics
        self.metrics_collector.record_evaluation_result(
            problem_id=problem_id,
            score=evaluation_results.get("math_score", 0),
            duration_ms=execution_time_ms,
            tags={
                "agent_system": self.name,
                "evaluator": self.evaluator_name,
                "problem_type": self.evaluator_name
            }
        )
        
        # Return final results
        return {
            **evaluation_results,  # Include all evaluation results
            "token_usage": token_usage,
            "execution_time_ms": execution_time_ms,
        }


# Register the agent system
AgentSystemRegistry.register("supervisor_mas", SupervisorMAS, evaluator="math")


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
    with open("benchmark/data/math_test.jsonl", "r") as f:
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
            all_results.append(
                {
                    "problem": problem["problem"],
                    "expected": problem["solution"],
                    "prediction": results["extracted_answer"],
                    "passed": results["math_score"] == 1,
                    "evaluation": results["run_evaluation"],
                    "token_usage": results["token_usage"],
                    "execution_time_ms": results["execution_time_ms"],
                }
            )

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

    print("\nSummary:")
    print(f"Passed: {passed}/{total} ({passed / total:.2%})")
    print(f"Total tokens: {total_tokens}")
    print(f"Average tokens per problem: {total_tokens / total:.2f}")
    print(f"Total execution time: {total_time:.2f}ms")
    print(f"Average execution time: {total_time / total:.2f}ms per problem")
