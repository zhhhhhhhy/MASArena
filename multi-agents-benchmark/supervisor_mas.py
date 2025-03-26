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
from ext.benchmark.math import MATHBenchmark

# root_path = Path(__file__).parent

# project_root = Path(__file__).parent.parent

# sys.path.append(str(root_path))


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

    # system_prompt = """You are a mathematical problem solver.
    # Always provide your final answer in LaTeX format using \\boxed{}.
    # For example: \\boxed{42} or \\boxed{x^2 + 2x}"""

    coder_agent = create_react_agent(
        model,
        tools=[],
        # prompt=system_prompt
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
    builder = StateGraph(State)
    checkpointer = InMemorySaver()

    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", create_supervisor())
    builder.add_node("researcher", create_research_node())
    builder.add_node("coder", create_code_node())

    return builder.compile(checkpointer=checkpointer)


def evaluate_mas(problem: dict):
    run_evaluator = RunEvaluator()
    math_evaluator = MATHBenchmark(
        name="MATH",
        file_path="multi-agents-benchmark/ext/data/math_test.jsonl",
        log_path="multi-agents-benchmark/ext/data/results/MATH",
    )

    graph = create_mas_graph()
    initial_state = {
        "messages": [("user", problem["problem"])],
        "token_usage": {"supervisor": 0, "researcher": 0, "coder": 0},
    }

    # 使用 invoke 获取最终状态
    run_result = graph.invoke(initial_state, config={"configurable": {"thread_id": "1"}})

    # 从最终状态中获取最后一条消息
    all_messages = run_result.get("messages", [])
    final_answer = ""

    # 正确处理消息对象
    if all_messages:
        last_msg = all_messages[-1]
        # 如果是元组格式 (role, content)
        if isinstance(last_msg, tuple) and len(last_msg) > 1:
            final_answer = last_msg[1]
        # 如果是 HumanMessage 对象
        elif hasattr(last_msg, "content"):
            final_answer = last_msg.content
        # 如果是字典格式 {'role': ..., 'content': ...}
        elif isinstance(last_msg, dict) and "content" in last_msg:
            final_answer = last_msg["content"]

    print(f"Final answer: {final_answer[:100]}...")

    score, extracted_answer = math_evaluator.calculate_score(problem["solution"], final_answer)

    run = Run(
        id=str(uuid.uuid4()),
        name="MATH_MAS_Evaluation",
        inputs={"problem": problem["problem"]},
        outputs={
            "prediction": final_answer,
            "extracted_answer": extracted_answer,
            "expected": problem["solution"],
            "math_score": score,
            "passed": score == 1,
        },
        run_type="evaluation",
        start_time="2025-03-11T12:00:00Z",
        trace_id=str(uuid.uuid4()),
    )

    run_evaluation = run_evaluator.evaluate_run(run=run)

    return {
        "final_answer": final_answer,
        "math_score": score,
        "run_evaluation": run_evaluation,
        "extracted_answer": extracted_answer,
    }


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
    import json

    with open("multi-agents-benchmark/ext/data/math_test.jsonl", "r") as f:
        problems = [json.loads(line) for line in f]

    # 记录所有结果
    all_results = []

    # 每次只处理一个问题
    for i, problem in enumerate(problems):
        print(f"\nProcessing Problem {i + 1}/{len(problems)}")
        print(f"Problem: {problem['problem'][:100]}...")

        try:
            # 每个问题创建新的评估实例
            results = evaluate_mas(problem)

            # 保存这个问题的结果
            all_results.append(
                {
                    "problem": problem["problem"],
                    "expected": problem["solution"],
                    "prediction": results["extracted_answer"],
                    "passed": results["math_score"] == 1,
                    "evaluation": results["run_evaluation"],
                }
            )

            print(f"Expected: {problem['solution']}")
            print(f"Predicted: {results['extracted_answer']}")
            print(f"Passed: {results['math_score'] == 1}")

        except Exception as e:
            print(f"Error processing problem {i + 1}: {str(e)}")
            continue

    # 打印总体统计
    total = len(problems)
    passed = sum(1 for r in all_results if r["passed"])
    print(f"Passed: {passed}")
    print(f"Accuracy: {passed / total:.2%}")
