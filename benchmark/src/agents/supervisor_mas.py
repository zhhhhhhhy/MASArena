"""
Supervisor-based Multi-Agent System

This module implements a supervisor-based multi-agent system where a supervisor
agent coordinates the work of specialized agents.
"""

import os
from typing import Literal, Dict, TypedDict, Any
import time
import uuid

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage
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

load_dotenv()


class State(MessagesState):
    next: str


class Router(TypedDict):
    next: Literal["researcher", "coder", "FINISH"]


@traceable
def create_supervisor():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
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

        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


@traceable
def create_research_node():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
    )
    research_agent = create_react_agent(model, tools=[])

    def research_node(state: State) -> Command[Literal["supervisor"]]:
        with get_openai_callback() as cb:
            result = research_agent.invoke(state)

        ai_message = result["messages"][-1]
        ai_message.name = "researcher"
        return Command(
            update={
                "messages": [ai_message],
            },
            goto="supervisor",
        )

    return research_node


@traceable
def create_code_node():
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL"),
    )

    coder_agent = create_react_agent(
        model,
        tools=[],
    )

    def code_node(state: State) -> Command[Literal["supervisor"]]:
        with get_openai_callback() as cb:
            result = coder_agent.invoke(state)

        ai_message = result["messages"][-1]
        ai_message.name = "coder"
        return Command(
            update={
                "messages": [ai_message],
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
        
        # Initialize evaluator and metrics collector through base class methods
        self._initialize_evaluator()
        self._initialize_metrics_collector()

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a problem without evaluation.
        
        This method focuses on running the agent system and collecting basic metrics.
        
        Args:
            problem: Dictionary containing the problem data
            
        Returns:
            Dictionary with run results (messages)
        """
        # Create graph
        graph = create_mas_graph()
        
        # Create initial state
        initial_state = {
            "messages": [("user", problem["problem"])],
        }

        # Run the graph
        run_result = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
        
        # Return results directly
        return {
            "messages": run_result.get("messages", []),
        }


# Register the agent system
AgentSystemRegistry.register("supervisor_mas", SupervisorMAS, evaluator="math")
