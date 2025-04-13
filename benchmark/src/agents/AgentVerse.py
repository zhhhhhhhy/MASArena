import time
import json
import os
import asyncio
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Literal, Dict, TypedDict, Any
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Annotated, List, Tuple, TypedDict
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.callbacks.manager import AsyncCallbackManager
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run
from langgraph.graph import StateGraph, START, END, MessagesState
from src.evaluators.math import MATHBenchmark
from src.agents.base import AgentSystem, AgentSystemRegistry

@dataclass
class ExpertProfile:
    id: str
    name: str
    description: str

class agent(BaseModel):
    name: str
    describe: str
    agent_id: int

class Agents(BaseModel):
    agents: list[agent]


class Discussion(TypedDict):
    agent_id: int
    context: str
    token_usage: Any

class SumDiscussion(TypedDict):
    sum_context: list[Discussion]

dis_graph = SumDiscussion(
    sum_context=None
)

class RecruiterAgent:
    def __init__(self, agent_id: str, model_name: str = None, system_prompt: str = None):
        self.agent_id = agent_id
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            "You are a professional AI recruitment expert who needs to generate the right work team configuration based on the needs of the problem."
            """Please strictly follow the following rules:
            1.Generate expert descriptions in different fields at a time,
            2.Output in dict format, structure must contain agents array
            3.Each expert contains 2 fields: name, describe
            4.The description includes the specific division of labor needed to solve the problem"""
        )
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        ).with_structured_output(Agents)
        self.token_usage = 0

    def _create_prompt(self, problem: str, num: int = 3,) -> str:
        return f"""
            Generate the configuration of {num} expert agents based on the following problem requirements:

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
            Think carefully about the problem step by step.Describe in detail the roles of different experts



    Agent ID: {self.agent_id}
    """

    def describe(self, problem: str):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_prompt(problem))
        ]

        start_time = time.time()
        with get_openai_callback() as cb:
            response = self.llm.invoke(messages)
        end_time = time.time()

        self.token_usage = cb.total_tokens
        return {
            "agent_id": self.agent_id,
            "solution": response.model_dump(),
            "token_usage": self.token_usage,
            "latency_ms": (end_time - start_time) * 1000,
        }

class WorkAgents:
    def __init__(self, agent_id: str, system_prompt: str = None) -> object:
        self.agent_id = agent_id
        self.model_name = "gpt-4o-mini"
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
        self.token_usage = 0


class AgentVerse(AgentSystem):
    MAX_EVALUATE_LENGTH = 3000  # 根据模型上下文窗口调整
    def __init__(self, name: str = "agentverse", config: Dict[str, Any] = None):
        """Initialize the Swarm Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        self.evaluator_name = self.config.get("evaluator", "math")
        self.num_agents = self.config.get("num_agents", 3)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.use_parallel = self.config.get("parallel", True)
        self.token_usage = {}

    def _create_agents(self, problem):
        recruiter = RecruiterAgent(
            agent_id="HR_Manager_001",
            model_name="gpt-4o-mini",
        )
        response_dict = recruiter.describe(problem)
        agents_list = response_dict.get("solution", {}).get("agents", [])
        expert_team = [
            ExpertProfile(
                id=agent.get("agent_id", "000"),  # 保证ID为三位数
                name=agent.get("name", "未命名专家").strip(),
                description=agent.get("describe", "")[:500]  # 截断过长的描述
            ) for agent in agents_list
            if isinstance(agent, dict)  # 类型检查（网页8的数据验证原则）
        ]
        print(expert_team, "\n")
        Workers = []
        for expert in expert_team:
            Workers.append(
                WorkAgents(
                    agent_id=expert.id,
                    system_prompt=expert.description
                )
            )
        return Workers

    async def inner_evaluate(self, solution: str):
        class Eva(BaseModel):
            evaluation: str
            score: int = Field(ge=0, le=100)

        # processed_solution = self._preprocess_solution(solution)

        llm = ChatOpenAI(
            model = "gpt-4o-mini",
            base_url = os.getenv("BASE_URL"),
            api_key = os.getenv("OPENAI_API_KEY"),
            max_retries=3,
            max_tokens=1000
        ).with_structured_output(Eva)

        prompt = f"""Please generate an overall short evaluation and improvement suggestions based on the discussion below, with a rating of 0-100.

                discussion：
                {solution}

                return：
                {{
                    "evaluation": "evaluation text",
                    "score": "evaluation score"
                }}"""

        print("inner_evaluate start...")

        # try:
        #     evaluation = await llm.ainvoke(prompt)
        #     print(evaluation)
        #     return evaluation
        # except Exception as e:
        #     print(f"评估请求失败: {str(e)}")
        #     return Eva(evaluation="评估失败", score=0)
        evaluation = Eva(evaluation= "No" ,score = 90)
        return evaluation

    def output(self, raw_solution):
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        print("output start...")
        final_solution = llm.invoke(
            f"According to the above conclusions：{raw_solution}.the final solution is summarized(Only Answer)")
        print(final_solution)
        return str(final_solution)

    async def async_worker(self, worker: Any, problem: str, feedback: str = None) -> Discussion:
        # 创建异步回调处理器
        callback_handler = OpenAICallbackHandler()
        async_manager = AsyncCallbackManager([callback_handler])

        # 构造消息（需包含系统提示）
        messages = [
            SystemMessage(content=worker.system_prompt),
            HumanMessage(content=problem + (feedback or ""))
        ]

        # 异步调用并捕获Token
        response = await worker.llm.ainvoke(
            messages,
            config={"callbacks": async_manager}
        )

        # 更新Token统计
        total_tokens = callback_handler.total_tokens
        worker.token_usage += total_tokens

        return Discussion(
            agent_id=worker.agent_id,
            context=response.content,
            token_usage=total_tokens  # 本次调用的Token数
        )


    async def async_solve_problem(self, problem: str, workers: list, feedback: str = None) -> list[Discussion]:
        print("async_solve_problem start...")
        # 创建协程任务列表
        tasks = [
            asyncio.create_task(
                self.async_worker(worker, problem, feedback)
            ) for worker in workers
        ]

        # 并发执行并收集结果
        solutions = await asyncio.gather(*tasks)

        # 汇总Token统计到主系统
        for worker in workers:
            self.token_usage[worker.agent_id] = worker.token_usage  # 累加或覆盖

        print("Solutions:", solutions, "\n")
        return solutions

    async def solve_problem_with_feedback(self, problem_text: str, agents: list):
        feedback = None
        max_attempts = 5  # 防止无限循环
        for attempt in range(max_attempts):
            # 异步获取解决方案
            solutions = await self.async_solve_problem(problem_text, agents, feedback)

            # 将讨论内容合并为字符串
            raw_solution = "\n".join([s['context'] for s in solutions])
            print(raw_solution)

            # 异步评估解决方案
            evaluation_task = asyncio.create_task(self.inner_evaluate(raw_solution))

            evaluation = await evaluation_task

            # 检查评分
            if evaluation.score >= 80:
                print(f"方案在尝试 {attempt + 1} 次后达到标准")
                return self.output(raw_solution)

            feedback = evaluation.get('evaluation', '')

        # 达到最大尝试次数仍不合格
        print(f"经过 {max_attempts} 次尝试仍未达标，返回最终方案")
        return self.output(raw_solution)

    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Evaluate the agent system on a problem"""
        metrics_registry = kwargs.get("metrics_registry", self.metrics_registry)
        problem_text = problem["problem"]
        run_evaluator = RunEvaluator()
        math_evaluator = MATHBenchmark(
            name=self.evaluator_name.upper(),
            file_path=f"benchmark/data/{self.evaluator_name}_test.jsonl",
            log_path=f"benchmark/data/results/{self.evaluator_name.upper()}",
        )

        # Record start time
        start_time = time.time()

        # Create swarm agents
        agents = self._create_agents(problem_text)

        # Collect agent solutions
        agent_solutions = []
        token_usage = {}


        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            final_answer = loop.run_until_complete(
                self.solve_problem_with_feedback(problem_text, agents)
            )
        finally:
            loop.close()

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
            "token_usage": self.token_usage,  # 返回实例的token_usage属性
            "execution_time_ms": execution_time_ms,
            "agent_solutions": [{"agent_id": s["agent_id"], "solution": s["solution"]} for s in agent_solutions],
        }

# Register the agent system
AgentSystemRegistry.register("AgentVerse", AgentVerse, evaluator="math", num_agents=3, parallel=True)

if __name__ == "__main__":
    # Test the agent system
    with open("benchmark/data/math_test.jsonl", "r") as f:
        problems = [json.loads(line) for line in f]

    # Process only a single problem for testing
    test_problem = problems[0]
    print(f"Problem: {test_problem['problem'][:100]}...")

    # Create and run the swarm
    agentverse = AgentVerse(config={"num_agents": 2})
    results = agentverse.evaluate(test_problem)

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









