import sys
from pathlib import Path
from typing import Literal, Dict, TypedDict
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
import uuid
import os
import asyncio
# from multi-agents-benchmark.ext.benchmark.math import MATHBenchmark

root_path = Path(__file__).parent.parent.parent / "multi-agents-benchmark"
sys.path.append(str(root_path))


load_dotenv()

tavily_tool = TavilySearchResults(max_results=5)


class State(MessagesState):
    next: str
    token_usage: Dict[str, int]


class Router(TypedDict):
    next: Literal["researcher", "coder", "FINISH"]


@traceable
def create_supervisor():
    model = ChatOpenAI(model="gpt-4o-mini")
    members = ["researcher", "coder"]

    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )

    def supervisor_node(state: State) -> Command[Literal["researcher", "coder", "__end__"]]:
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]

        with get_openai_callback() as cb:
            response = model.with_structured_output(Router).invoke(messages)

        state["token_usage"]["supervisor"] += cb.total_tokens
        goto = response["next"]

        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto, "token_usage": state["token_usage"]})

    return supervisor_node


@traceable
def create_research_node():
    model = ChatOpenAI(model="gpt-4o-mini")
    research_agent = create_react_agent(model, tools=[tavily_tool])

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
    model = ChatOpenAI(model="gpt-4o-mini")
    coder_agent = create_react_agent(model, tools=[])

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
    builder = StateGraph(State)
    checkpointer = InMemorySaver()

    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", create_supervisor())
    builder.add_node("researcher", create_research_node())
    builder.add_node("coder", create_code_node())

    return builder.compile(checkpointer=checkpointer)


def evaluate_mas(query: str):
    evaluator = RunEvaluator()
    graph = create_mas_graph()
    initial_state = {"messages": [("user", query)], "token_usage": {"supervisor": 0, "researcher": 0, "coder": 0}}

    results = []
    for step in graph.stream(initial_state, config={"configurable": {"thread_id": "1"}}):
        results.append(
            {
                "step": len(results) + 1,
                "agent": step.get("next", "unknown"),
                "messages": step.get("messages", []),
                "token_usage": step.get("token_usage", {}),
            }
        )
        print(f"Step {len(results)}: {step}")

    run = Run(
        id=str(uuid.uuid4()),
        name="MAS_Evaluation",
        inputs={"query": query},
        outputs={"results": results},
        run_type="evaluation",
        start_time="2025-03-11T12:00:00Z",
        trace_id=str(uuid.uuid4()),
    )

    evaluation = evaluator.evaluate_run(run=run)

    return results, evaluation


# class MASEvaluator(DROPBenchmark):
#     def __init__(self):
#         super().__init__(
#             name="DROP",
#             file_path="../../multi-agents-benchmark/ext/data/drop_test.jsonl",
#             log_path="../../multi-agents-benchmark/ext/data/results/DROP"
#         )

#     def calculate_metrics(self, run: RunTree, expected_output: str) -> dict:
#         prediction = run.outputs["output"]
#         f1_score, _ = self.calculate_score(expected_output, prediction)
#         return {
#             "f1_score": f1_score,
#             "exact_match": f1_score == 1.0
#         }

#     async def _generate_output(self, chain, input_text):
#         result = await chain.ainvoke({"input": input_text})
#         cost = sum(run.cost for run in result.runs) if hasattr(result, "runs") else 0
#         return result["output"], cost


if __name__ == "__main__":
    # query = "What is the GDP of California and New York? Compute their average."
    query = "Give me the python code to calculate the sum of the first 100 natural numbers."
    results, evaluation = evaluate_mas(query)
    print("\nEvaluation Results:")
    print(evaluation)

    # evaluator = MASEvaluator()
    # asyncio.run(evaluator.run())
