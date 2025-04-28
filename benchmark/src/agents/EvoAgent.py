import time
import json
import os
import asyncio
import random
import uuid
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, TypedDict, Any, List, Optional, Tuple
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import Annotated
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
class Agent:
    """代表一个LLM代理"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    score: float = 0.0
    result: Dict[str, Any] = field(default_factory=dict)
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """解决给定问题并返回结果"""
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=problem)
        ]
        
        start_time = time.time()
        response = llm.invoke(messages)
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # 确保正确提取token使用情况
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if hasattr(response, "usage"):
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
        elif hasattr(response, "llm_output") and hasattr(response.llm_output, "token_usage"):
            prompt_tokens = response.llm_output.token_usage.prompt_tokens
            completion_tokens = response.llm_output.token_usage.completion_tokens
            total_tokens = response.llm_output.token_usage.total_tokens
        
        result = {
            "agent_id": self.agent_id,
            "name": self.name,
            "execution_time_ms": execution_time_ms,
            "extracted_answer": response.content,
            "llm_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        
        self.result = result
        return result

class EvoAgent(AgentSystem):
    """
    基于进化算法的多智能体系统
    
    算法流程:
    1. 初始化3个基础智能体
    2. 第一次迭代: 交叉操作，根据父代智能体的结果和初始智能体，更新父代智能体的设置，生成新的子代智能体
    3. 第二次迭代: 变异操作，基于父代智能体和初始智能体，生成更多的子代智能体
    4. 筛选出5个最佳智能体
    5. 将问题投给最终的五个智能体，分别生成回答
    6. 通过新的LLM汇总输出
    """
    
    def __init__(self, name: str = "evoagent", config: Dict[str, Any] = None):
        """
        初始化进化智能体系统
        
        Args:
            name: 系统名称
            config: 配置参数
        """
        super().__init__(name, config)
        
        # 默认配置
        self.model_name = self.config.get("model_name", "gpt-4o-mini")
        self.initial_agents_count = self.config.get("initial_agents_count", 3)
        self.final_agents_count = self.config.get("final_agents_count", 5)
        self.crossover_rate = self.config.get("crossover_rate", 0.7)
        self.mutation_rate = self.config.get("mutation_rate", 0.3)
        
        # 初始化评估器和指标收集器
        self._initialize_evaluator()
        self._initialize_metrics_collector()
        
    def _initialize_base_agents(self) -> List[Agent]:
        """初始化基础智能体"""
        base_agents = []
        
        # 基础系统提示模板
        base_prompts = [
            "你是一个数学专家，擅长解决数学问题。请一步一步地思考并解决问题。",
            "你是一个逻辑推理专家，擅长分析问题并找出解决方案。请提供详细的推理过程。",
            "你是一个问题解决专家，擅长将复杂问题分解为简单步骤。请清晰地展示你的思考过程。"
        ]
        
        # 创建初始智能体
        for i in range(self.initial_agents_count):
            agent_id = str(uuid.uuid4())
            name = f"EVO-{i+1}"  # 使用EVO-1、EVO-2、EVO-3等格式
            system_prompt = base_prompts[i % len(base_prompts)]
            
            agent = Agent(
                agent_id=agent_id,
                name=name,
                model_name=self.model_name,
                system_prompt=system_prompt
            )
            
            base_agents.append(agent)
            
        return base_agents
    
    def _crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        交叉操作：结合两个父代智能体的特征创建子代
        
        Args:
            parent1: 第一个父代智能体
            parent2: 第二个父代智能体
            
        Returns:
            子代智能体
        """
        # 使用LLM进行交叉
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = f"""
        你正在执行两个AI智能体配置的交叉操作，以创建一个新的、改进的智能体。

        父代1配置:
        - 名称: {parent1.name}
        - 系统提示: {parent1.system_prompt}
        - 结果: {parent1.result.get('extracted_answer', '无结果')}

        父代2配置:
        - 名称: {parent2.name}
        - 系统提示: {parent2.system_prompt}
        - 结果: {parent2.result.get('extracted_answer', '无结果')}

        请创建一个新的智能体配置，结合两个父代的最佳特点。
        新配置应该继承两个父代的优点，同时避免它们的缺点。

        请以JSON格式返回新配置，包含以下字段:
        - name: 新智能体的名称 (必须以"EVO-C-"开头，例如"EVO-C-1")
        - system_prompt: 新智能体的系统提示
        """
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        try:
            # 尝试解析JSON响应
            config = json.loads(response.content)
            
            # 确保名称以EVO-C-开头
            name = config.get("name", "")
            if not name.startswith("EVO-C-"):
                name = f"EVO-C-{random.randint(1, 999)}"
            
            # 创建子代智能体
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=name,
                model_name=self.model_name,
                system_prompt=config.get("system_prompt", parent1.system_prompt)
            )
            
            return child
        except Exception as e:
            # 如果解析失败，使用简单的随机选择
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-C-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
            )
            
            return child
    
    def _mutation(self, parent: Agent) -> Agent:
        """
        变异操作：基于父代智能体创建变异的子代
        
        Args:
            parent: 父代智能体
            
        Returns:
            变异的子代智能体
        """
        # 使用LLM进行变异
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        prompt = f"""
        你正在对AI智能体配置执行变异操作，以创建一个变异的版本。

        父代配置:
        - 名称: {parent.name}
        - 系统提示: {parent.system_prompt}
        - 结果: {parent.result.get('extracted_answer', '无结果')}

        请创建一个变异的智能体配置，与父代不同但仍然有效。
        变异应该引入一些随机性，同时保持智能体解决问题的能力。

        请以JSON格式返回变异后的配置，包含以下字段:
        - name: 变异后智能体的名称 (必须以"EVO-M-"开头，例如"EVO-M-1")
        - system_prompt: 变异后智能体的系统提示
        """
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        try:
            # 尝试解析JSON响应
            config = json.loads(response.content)
            
            # 确保名称以EVO-M-开头
            name = config.get("name", "")
            if not name.startswith("EVO-M-"):
                name = f"EVO-M-{random.randint(1, 999)}"
            
            # 创建变异的子代智能体
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=name,
                model_name=self.model_name,
                system_prompt=config.get("system_prompt", parent.system_prompt)
            )
            
            return child
        except Exception as e:
            # 如果解析失败，使用简单的随机修改
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-M-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent.system_prompt + f" 变异版本 {random.randint(1, 100)}"
            )
            
            return child
    
    def _calculate_score(self, result: Dict[str, Any], problem: Dict[str, Any]) -> float:
        """
        计算结果的得分
        
        Args:
            result: 智能体的结果
            problem: 问题
            
        Returns:
            得分（0-1之间）
        """
        try:
            # 提取答案
            extracted_answer = result.get("extracted_answer", "")
            
            # 使用评估器计算得分
            score, _ = self.evaluator.calculate_score(problem.get("solution", ""), extracted_answer)
            
            return score
        except Exception as e:
            return 0.0
    
    def _summarize_results(self, problem: str, results: List[Dict[str, Any]]) -> str:
        """
        使用LLM汇总多个智能体的结果
        
        Args:
            problem: 问题
            results: 多个智能体的结果列表
            
        Returns:
            汇总后的结果
        """
        llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 构建汇总提示
        results_text = ""
        for i, result in enumerate(results):
            results_text += f"智能体 {i+1} ({result.get('name', f'Agent-{i+1}')}) 的回答:\n"
            results_text += f"{result.get('extracted_answer', '无回答')}\n\n"
        
        prompt = f"""
        请汇总以下多个智能体对同一问题的回答，生成一个综合的、全面的答案。

        问题:
        {problem}

        {results_text}

        请提供一个综合的答案，结合所有智能体的优点，并解决任何冲突或矛盾。
        你的回答应该是最完整、最准确的。
        """
        
        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # 记录token使用情况
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if hasattr(response, "usage"):
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
        elif hasattr(response, "llm_output") and hasattr(response.llm_output, "token_usage"):
            prompt_tokens = response.llm_output.token_usage.prompt_tokens
            completion_tokens = response.llm_output.token_usage.completion_tokens
            total_tokens = response.llm_output.token_usage.total_tokens
        
        # 将token使用情况添加到结果中
        for result in results:
            if "llm_usage" not in result:
                result["llm_usage"] = {}
            result["llm_usage"]["summary_prompt_tokens"] = prompt_tokens
            result["llm_usage"]["summary_completion_tokens"] = completion_tokens
            result["llm_usage"]["summary_total_tokens"] = total_tokens
        
        return response.content
    
    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        运行进化智能体系统解决给定问题
        
        Args:
            problem: 问题字典
            **kwargs: 额外参数
            
        Returns:
            结果字典
        """
        # 记录开始时间
        start_time = time.time()
        
        # 提取问题文本
        problem_text = problem.get("problem", "")
        
        # 初始化基础智能体
        base_agents = self._initialize_base_agents()
        
        # 运行基础智能体获取初始结果
        for agent in base_agents:
            result = agent.solve(problem_text)
            score = self._calculate_score(result, problem)
            agent.score = score
            agent.result = result
        
        # 按得分排序基础智能体
        base_agents.sort(key=lambda x: x.score, reverse=True)
        
        # 第一次迭代：交叉操作
        crossover_agents = []
        
        # 保留最佳基础智能体
        crossover_agents.append(base_agents[0])
        
        # 创建新的交叉智能体
        while len(crossover_agents) < self.initial_agents_count * 2:
            # 随机选择两个父代
            parent1 = random.choice(base_agents)
            parent2 = random.choice(base_agents)
            
            # 执行交叉
            child = self._crossover(parent1, parent2)
            crossover_agents.append(child)
        
        # 运行交叉智能体
        for agent in crossover_agents[1:]:  # 跳过已评估的最佳基础智能体
            result = agent.solve(problem_text)
            score = self._calculate_score(result, problem)
            agent.score = score
            agent.result = result
        
        # 按得分排序交叉智能体
        crossover_agents.sort(key=lambda x: x.score, reverse=True)
        
        # 第二次迭代：变异操作
        mutation_agents = []
        
        # 保留最佳交叉智能体
        mutation_agents.append(crossover_agents[0])
        
        # 创建新的变异智能体
        while len(mutation_agents) < self.initial_agents_count * 3:
            # 随机选择一个父代
            parent = random.choice(crossover_agents)
            
            # 执行变异
            child = self._mutation(parent)
            mutation_agents.append(child)
        
        # 运行变异智能体
        for agent in mutation_agents[1:]:  # 跳过已评估的最佳交叉智能体
            result = agent.solve(problem_text)
            score = self._calculate_score(result, problem)
            agent.score = score
            agent.result = result
        
        # 按得分排序所有智能体
        mutation_agents.sort(key=lambda x: x.score, reverse=True)
        
        # 选择最终的5个最佳智能体
        final_agents = mutation_agents[:self.final_agents_count]
        
        # 汇总最终智能体的结果
        final_results = [agent.result for agent in final_agents]
        summary = self._summarize_results(problem_text, final_results)
        
        # 记录结束时间
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # 构建最终结果
        result = {
            "execution_time_ms": execution_time_ms,
            "extracted_answer": summary,
            "final_agents": [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "score": agent.score,
                    "answer": agent.result.get("extracted_answer", "")
                }
                for agent in final_agents
            ],
            "evolution_metrics": {
                "initial_agents": len(base_agents),
                "crossover_agents": len(crossover_agents),
                "mutation_agents": len(mutation_agents),
                "final_agents": len(final_agents),
                "best_score": final_agents[0].score if final_agents else 0.0
            }
        }
        
        # 添加messages字段，用于记录tokens和回答
        messages = []
        
        # 添加所有智能体的回答作为消息
        for agent in final_agents:
            # 创建一个AIMessage对象，包含智能体的回答和token使用情况
            from langchain_core.messages import AIMessage
            
            # 从agent.result中提取token使用情况
            llm_usage = agent.result.get("llm_usage", {})
            
            # 创建usage_metadata字典
            usage_metadata = {
                "input_tokens": llm_usage.get("prompt_tokens", 0),
                "output_tokens": llm_usage.get("completion_tokens", 0),
                "total_tokens": llm_usage.get("total_tokens", 0),
                "output_token_details": {"reasoning": 0}  # 默认值
            }
            
            # 创建AIMessage对象
            ai_message = AIMessage(
                content=agent.result.get("extracted_answer", ""),
                name=agent.name,
                id=agent.agent_id,
                usage_metadata=usage_metadata
            )
            
            messages.append(ai_message)
        
        # 添加汇总结果作为消息
        from langchain_core.messages import AIMessage
        
        # 计算汇总消息的token使用情况（简单估算）
        total_prompt_tokens = sum(agent.result.get("llm_usage", {}).get("prompt_tokens", 0) for agent in final_agents)
        total_completion_tokens = sum(agent.result.get("llm_usage", {}).get("completion_tokens", 0) for agent in final_agents)
        
        # 创建汇总消息的usage_metadata
        summary_usage_metadata = {
            "input_tokens": total_prompt_tokens,
            "output_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "output_token_details": {"reasoning": 0}  # 默认值
        }
        
        # 创建汇总消息
        summary_message = AIMessage(
            content=summary,
            name="EVO-SUMMARY",  # 使用EVO-SUMMARY格式
            id="summary",
            usage_metadata=summary_usage_metadata
        )
        
        messages.append(summary_message)
        
        # 将messages添加到结果中
        result["messages"] = messages
        
        return result

# 注册智能体系统
AgentSystemRegistry.register("evoagent", EvoAgent)

if __name__ == "__main__":
    # 测试EvoAgent
    problem = {
        "problem": "一个正整数，它的平方根是 452，求这个正整数。"
    }
    result = EvoAgent().run_agent(problem)
    print(result)
