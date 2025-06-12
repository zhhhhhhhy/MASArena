import time
import json
import os
import asyncio
import random
import uuid
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Dict, TypedDict, Any, List, Optional, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

import nest_asyncio
nest_asyncio.apply()

# 禁用LangSmith跟踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# Load environment variables
load_dotenv()

# 添加颜色输出支持
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(step_name: str, color: str = Colors.BLUE):
    """打印步骤信息"""
    print(f"\n{color}{Colors.BOLD}===== {step_name} ====={Colors.ENDC}")

def print_agent_info(agent: 'Agent', score: float = None):
    """打印智能体信息"""
    score_str = f" (得分: {score:.4f})" if score is not None else ""
    print(f"{Colors.CYAN}智能体: {agent.name}{score_str}{Colors.ENDC}")
    print(f"  系统提示: {agent.system_prompt[:100]}...")

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
    llm: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        """Initialize LLM after dataclass init."""
        self.llm = ChatOpenAI(model=self.model_name)
    
    async def solve(self, problem: str) -> Dict[str, Any]:
        """解决给定问题并返回结果"""
        # 创建回调处理器来收集token使用情况
        callback_handler = OpenAICallbackHandler()
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=problem)
        ]
        
        start_time = time.time()
        response = await self.llm.ainvoke(messages, config={'callbacks': [callback_handler]})
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        
        # 从回调处理器获取token使用情况
        input_tokens = callback_handler.prompt_tokens
        output_tokens = callback_handler.completion_tokens
        total_tokens = callback_handler.total_tokens
        
        # 为AIMessage添加usage_metadata
        if isinstance(response, AIMessage):
            response.usage_metadata = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_token_details": {
                    "system_prompt": len(self.system_prompt.split()),
                    "user_prompt": len(problem.split())
                },
                "output_token_details": {
                    "reasoning": output_tokens,  # 简化处理，将所有输出标记视为推理
                }
            }
        
        result = {
            "agent_id": self.agent_id,
            "name": self.name,
            "execution_time_ms": execution_time_ms,
            "extracted_answer": response.content,
            "usage_metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
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
    
    async def _crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """
        交叉操作：结合两个父代智能体的特征创建子代
        
        Args:
            parent1: 第一个父代智能体
            parent2: 第二个父代智能体
            
        Returns:
            子代智能体
        """
        try:
            # 添加超时处理
            async with asyncio.timeout(30):  # 设置30秒超时
                # 使用LLM进行交叉，添加回调以收集token用量
                callback_handler = OpenAICallbackHandler()
                
                llm = ChatOpenAI(
                    model=self.model_name
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

                请确保返回的是有效的JSON格式，不要添加任何额外的文本或解释。
                """
                
                response = await llm.ainvoke([{"role": "user", "content": prompt}], config={'callbacks': [callback_handler]})
                
                # 添加token用量元数据
                if isinstance(response, AIMessage):
                    response.usage_metadata = {
                        "input_tokens": callback_handler.prompt_tokens,
                        "output_tokens": callback_handler.completion_tokens,
                        "total_tokens": callback_handler.total_tokens,
                        "input_token_details": {"prompt": len(prompt.split())},
                        "output_token_details": {"reasoning": callback_handler.completion_tokens}
                    }
                
                try:
                    # 尝试提取JSON内容
                    content = response.content.strip()
                    
                    # 尝试找到JSON开始和结束的位置
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        print(f"{Colors.CYAN}提取的JSON: {json_str[:100]}...{Colors.ENDC}")
                        
                        # 尝试解析JSON响应
                        config = json.loads(json_str)
                        
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
                    else:
                        print(f"{Colors.YELLOW}警告: 无法在响应中找到JSON: {content[:100]}...{Colors.ENDC}")
                        raise ValueError("无法在响应中找到JSON")
                except Exception as e:
                    print(f"{Colors.YELLOW}警告: 解析交叉结果失败: {str(e)}，使用简单随机选择{Colors.ENDC}")
                    # 如果解析失败，使用简单的随机选择
                    child = Agent(
                        agent_id=str(uuid.uuid4()),
                        name=f"EVO-C-{random.randint(1, 999)}",
                        model_name=self.model_name,
                        system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
                    )
                    
                    return child
        except asyncio.TimeoutError:
            print(f"{Colors.RED}警告: 交叉操作超时，使用简单随机选择{Colors.ENDC}")
            # 超时情况下使用简单的随机选择
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-C-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
            )
            
            return child
        except Exception as e:
            print(f"{Colors.RED}警告: 交叉操作出错: {str(e)}，使用简单随机选择{Colors.ENDC}")
            # 出错情况下使用简单的随机选择
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-C-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent1.system_prompt if random.random() < 0.5 else parent2.system_prompt
            )
            
            return child
    
    async def _mutation(self, parent: Agent) -> Agent:
        """
        变异操作：基于父代智能体创建变异的子代
        
        Args:
            parent: 父代智能体
            
        Returns:
            变异的子代智能体
        """
        try:
            # 添加超时处理
            async with asyncio.timeout(30):  # 设置30秒超时
                # 使用LLM进行变异，添加回调以收集token用量
                callback_handler = OpenAICallbackHandler()
                
                llm = ChatOpenAI(
                    model=self.model_name
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

                请确保返回的是有效的JSON格式，不要添加任何额外的文本或解释。
                """
                
                response = await llm.ainvoke([{"role": "user", "content": prompt}], config={'callbacks': [callback_handler]})
                
                # 添加token用量元数据
                if isinstance(response, AIMessage):
                    response.usage_metadata = {
                        "input_tokens": callback_handler.prompt_tokens,
                        "output_tokens": callback_handler.completion_tokens,
                        "total_tokens": callback_handler.total_tokens,
                        "input_token_details": {"prompt": len(prompt.split())},
                        "output_token_details": {"reasoning": callback_handler.completion_tokens}
                    }
                
                try:
                    # 尝试提取JSON内容
                    content = response.content.strip()
                    
                    # 尝试找到JSON开始和结束的位置
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        print(f"{Colors.CYAN}提取的JSON: {json_str[:100]}...{Colors.ENDC}")
                        
                        # 尝试解析JSON响应
                        config = json.loads(json_str)
                        
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
                    else:
                        print(f"{Colors.YELLOW}警告: 无法在响应中找到JSON: {content[:100]}...{Colors.ENDC}")
                        raise ValueError("无法在响应中找到JSON")
                except Exception as e:
                    print(f"{Colors.YELLOW}警告: 解析变异结果失败: {str(e)}，使用简单随机修改{Colors.ENDC}")
                    # 如果解析失败，使用简单的随机修改
                    child = Agent(
                        agent_id=str(uuid.uuid4()),
                        name=f"EVO-M-{random.randint(1, 999)}",
                        model_name=self.model_name,
                        system_prompt=parent.system_prompt + f" 变异版本 {random.randint(1, 100)}"
                    )
                    
                    return child
        except asyncio.TimeoutError:
            print(f"{Colors.RED}警告: 变异操作超时，使用简单随机修改{Colors.ENDC}")
            # 超时情况下使用简单的随机修改
            child = Agent(
                agent_id=str(uuid.uuid4()),
                name=f"EVO-M-{random.randint(1, 999)}",
                model_name=self.model_name,
                system_prompt=parent.system_prompt + f" 变异版本 {random.randint(1, 100)}"
            )
            
            return child
        except Exception as e:
            print(f"{Colors.RED}警告: 变异操作出错: {str(e)}，使用简单随机修改{Colors.ENDC}")
            # 出错情况下使用简单的随机修改
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
        except Exception:
            return 0.0
    
    async def _summarize_results(self, problem: str, results: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        使用LLM汇总多个智能体的结果
        
        Args:
            problem: 问题
            results: 多个智能体的结果列表
            
        Returns:
            汇总后的结果和token使用情况
        """
        try:
            # 添加超时处理
            async with asyncio.timeout(60):  # 设置60秒超时
                # 添加回调以收集token用量
                callback_handler = OpenAICallbackHandler()
                
                llm = ChatOpenAI(
                    model=self.model_name
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
                Remember the following rules:
                {self.format_prompt}
                """
                
                response = await llm.ainvoke([{"role": "user", "content": prompt}], config={'callbacks': [callback_handler]})
                
                # 创建token使用情况元数据
                usage_metadata = {
                    "input_tokens": callback_handler.prompt_tokens,
                    "output_tokens": callback_handler.completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                    "input_token_details": {"prompt": len(prompt.split())},
                    "output_token_details": {"reasoning": callback_handler.completion_tokens}
                }
                
                # 添加token用量元数据到AIMessage
                if isinstance(response, AIMessage):
                    response.usage_metadata = usage_metadata
                
                return response.content, usage_metadata
        except asyncio.TimeoutError:
            print(f"{Colors.RED}警告: 汇总结果超时，使用最佳智能体的回答{Colors.ENDC}")
            # 超时情况下使用最佳智能体的回答
            best_result = max(results, key=lambda x: x.get("score", 0))
            return best_result.get("extracted_answer", "无法汇总结果，使用最佳智能体的回答"), {}
        except Exception as e:
            print(f"{Colors.RED}警告: 汇总结果出错: {str(e)}，使用最佳智能体的回答{Colors.ENDC}")
            # 出错情况下使用最佳智能体的回答
            best_result = max(results, key=lambda x: x.get("score", 0))
            return best_result.get("extracted_answer", f"无法汇总结果: {str(e)}"), {}
    
    async def _run_agent_async(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        运行进化智能体系统解决给定问题（异步版本）
        
        Args:
            problem: 问题字典
            **kwargs: 额外参数
            
        Returns:
            结果字典
        """
        # 记录开始时间
        start_time = time.time()
        
        # 提取问题文本和问题ID
        problem_text = problem.get("problem", "")
        
        # 显示问题
        print_step("问题", Colors.GREEN)
        print(f"{Colors.YELLOW}{problem_text}{Colors.ENDC}")
        
        # 初始化基础智能体
        print_step("初始化基础智能体")
        base_agents = self._initialize_base_agents()
        for agent in base_agents:
            print_agent_info(agent)
        
        # 异步运行基础智能体获取初始结果
        print_step("运行基础智能体")
        tasks = []
        for agent in base_agents:
            tasks.append(self._run_agent_task(agent, problem_text, problem))
        
        # 使用简单的进度显示
        print(f"{Colors.CYAN}基础智能体进度: 0/{len(tasks)}{Colors.ENDC}")
        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                await task
            except Exception as e:
                print(f"{Colors.RED}警告: 任务执行出错: {str(e)}{Colors.ENDC}")
            completed += 1
            print(f"{Colors.CYAN}基础智能体进度: {completed}/{len(tasks)}{Colors.ENDC}")
        
        # 按得分排序基础智能体
        base_agents.sort(key=lambda x: x.score, reverse=True)
        
        # 显示基础智能体结果
        print_step("基础智能体结果", Colors.GREEN)
        for agent in base_agents:
            print_agent_info(agent, agent.score)
            print(f"  回答: {agent.result.get('extracted_answer', '')[:100]}...")
        
        # 第一次迭代：交叉操作
        print_step("第一次迭代: 交叉操作")
        crossover_agents = []
        
        # 保留最佳基础智能体
        crossover_agents.append(base_agents[0])
        print(f"{Colors.GREEN}保留最佳基础智能体: {base_agents[0].name}{Colors.ENDC}")
        
        # 创建新的交叉智能体 - 异步并行执行
        print_step("创建交叉智能体")
        crossover_tasks = []
        for _ in range(self.initial_agents_count * 2 - 1):  # 减1是因为已经添加了一个最佳基础智能体
            # 随机选择两个父代
            parent1 = random.choice(base_agents)
            parent2 = random.choice(base_agents)
            
            # 异步执行交叉
            crossover_tasks.append(self._crossover(parent1, parent2))
        
        # 等待所有交叉任务完成
        crossover_results = await asyncio.gather(*crossover_tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(crossover_results):
            if isinstance(result, Exception):
                print(f"{Colors.RED}警告: 交叉任务 {i+1} 执行出错: {str(result)}{Colors.ENDC}")
                continue
                
            crossover_agents.append(result)
            print_agent_info(result)
            print("  父代: 随机选择")
        
        # 异步运行交叉智能体
        print_step("运行交叉智能体")
        tasks = []
        for agent in crossover_agents[1:]:  # 跳过已评估的最佳基础智能体
            tasks.append(self._run_agent_task(agent, problem_text, problem))
        
        # 使用简单的进度显示
        print(f"{Colors.CYAN}交叉智能体进度: 0/{len(tasks)}{Colors.ENDC}")
        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                await task
            except Exception as e:
                print(f"{Colors.RED}警告: 任务执行出错: {str(e)}{Colors.ENDC}")
            completed += 1
            print(f"{Colors.CYAN}交叉智能体进度: {completed}/{len(tasks)}{Colors.ENDC}")
        
        # 按得分排序交叉智能体
        crossover_agents.sort(key=lambda x: x.score, reverse=True)
        
        # 显示交叉智能体结果
        print_step("交叉智能体结果", Colors.GREEN)
        for agent in crossover_agents:
            print_agent_info(agent, agent.score)
            print(f"  回答: {agent.result.get('extracted_answer', '')[:100]}...")
        
        # 第二次迭代：变异操作
        print_step("第二次迭代: 变异操作")
        mutation_agents = []
        
        # 保留最佳交叉智能体
        mutation_agents.append(crossover_agents[0])
        print(f"{Colors.GREEN}保留最佳交叉智能体: {crossover_agents[0].name}{Colors.ENDC}")
        
        # 创建新的变异智能体 - 异步并行执行
        print_step("创建变异智能体")
        mutation_tasks = []
        for _ in range(self.initial_agents_count * 3 - 1):  # 减1是因为已经添加了一个最佳交叉智能体
            # 随机选择一个父代
            parent = random.choice(crossover_agents)
            
            # 异步执行变异
            mutation_tasks.append(self._mutation(parent))
        
        # 等待所有变异任务完成
        mutation_results = await asyncio.gather(*mutation_tasks, return_exceptions=True)
        
        # 处理结果
        for i, result in enumerate(mutation_results):
            if isinstance(result, Exception):
                print(f"{Colors.RED}警告: 变异任务 {i+1} 执行出错: {str(result)}{Colors.ENDC}")
                continue
                
            mutation_agents.append(result)
            print_agent_info(result)
            print("  父代: 随机选择")
        
        # 异步运行变异智能体
        print_step("运行变异智能体")
        tasks = []
        for agent in mutation_agents[1:]:  # 跳过已评估的最佳交叉智能体
            tasks.append(self._run_agent_task(agent, problem_text, problem))
        
        # 使用简单的进度显示
        print(f"{Colors.CYAN}变异智能体进度: 0/{len(tasks)}{Colors.ENDC}")
        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                await task
            except Exception as e:
                print(f"{Colors.RED}警告: 任务执行出错: {str(e)}{Colors.ENDC}")
            completed += 1
            print(f"{Colors.CYAN}变异智能体进度: {completed}/{len(tasks)}{Colors.ENDC}")
        
        # 按得分排序所有智能体
        mutation_agents.sort(key=lambda x: x.score, reverse=True)
        
        # 显示变异智能体结果
        print_step("变异智能体结果", Colors.GREEN)
        for agent in mutation_agents:
            print_agent_info(agent, agent.score)
            print(f"  回答: {agent.result.get('extracted_answer', '')[:100]}...")
        
        # 选择最终的5个最佳智能体
        print_step("选择最终智能体", Colors.GREEN)
        final_agents = mutation_agents[:self.final_agents_count]
        for agent in final_agents:
            print_agent_info(agent, agent.score)
        
        # 汇总最终智能体的结果
        print_step("汇总最终结果")
        final_results = [agent.result for agent in final_agents]
        summary, summary_usage = await self._summarize_results(problem_text, final_results)
        
        # 记录结束时间
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # 显示汇总结果
        print_step("最终汇总结果", Colors.GREEN)
        print(f"{Colors.YELLOW}{summary}{Colors.ENDC}")
        print(f"{Colors.CYAN}执行时间: {execution_time_ms:.2f}ms{Colors.ENDC}")
        
        # 构建消息列表，适配AgentSystem.evaluate的格式
        messages = []
        
        # 添加用户的原始问题
        user_message = HumanMessage(content=problem_text)
        messages.append(user_message)
        
        # 添加所有智能体的回答为AIMessage，并包含usage_metadata
        for agent in final_agents:
            ai_message = AIMessage(
                content=agent.result.get("extracted_answer", ""),
                name=agent.name
            )
            
            # 添加token使用情况元数据
            usage_metadata = agent.result.get("usage_metadata", {})
            if usage_metadata:
                ai_message.usage_metadata = {
                    "input_tokens": usage_metadata.get("input_tokens", 0),
                    "output_tokens": usage_metadata.get("output_tokens", 0),
                    "total_tokens": usage_metadata.get("total_tokens", 0),
                    "input_token_details": {
                        "system_prompt": len(agent.system_prompt.split()),
                        "user_prompt": len(problem_text.split())
                    },
                    "output_token_details": {
                        "reasoning": usage_metadata.get("output_tokens", 0)
                    }
                }
            
            messages.append(ai_message)
        
        # 添加最终汇总结果为AIMessage，并包含usage_metadata
        summary_message = AIMessage(
            content=summary,
            name="EVO-SUMMARY"
        )
        
        # 添加汇总的token使用情况元数据
        if summary_usage:
            summary_message.usage_metadata = summary_usage
            
        messages.append(summary_message)
        
        # 返回结果，包含消息、执行时间和进化指标
        return {
            "messages": messages,
            "final_answer": summary,
            "execution_time_ms": execution_time_ms,
            "evolution_metrics": {
                "initial_agents": len(base_agents),
                "crossover_agents": len(crossover_agents),
                "mutation_agents": len(mutation_agents),
                "final_agents": len(final_agents),
                "best_score": final_agents[0].score if final_agents else 0.0
            }
        }
        
    def run_agent_sync(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        同步版本的run_agent方法，用于兼容现有的同步代码
        
        Args:
            problem: 问题字典
            **kwargs: 额外参数
            
        Returns:
            结果字典
        """
        try:
            # Use asyncio.run() for robust async execution from a sync context
            return asyncio.run(self._run_agent_async(problem, **kwargs))
        except Exception as e:
            print(f"{Colors.RED}错误: 运行智能体时出错: {str(e)}{Colors.ENDC}")
            # 返回一个包含错误信息的结果
            return {
                "messages": [("error", f"执行出错: {str(e)}")],
                "execution_time_ms": 0
            }
            
    # 为了向后兼容，将run_agent_sync方法设置为run_agent的别名
    run_agent = run_agent_sync

    async def _run_agent_task(self, agent: Agent, problem_text: str, problem: Dict[str, Any]) -> None:
        """
        异步运行单个智能体并计算得分
        
        Args:
            agent: 要运行的智能体
            problem_text: 问题文本
            problem: 问题字典
        """
        try:
            # 添加超时处理
            async with asyncio.timeout(60):  # 设置60秒超时
                result = await agent.solve(problem_text)
                score = self._calculate_score(result, problem)
                agent.score = score
                agent.result = result
        except asyncio.TimeoutError:
            print(f"{Colors.RED}警告: 智能体 {agent.name} 执行超时{Colors.ENDC}")
            agent.score = 0.0
            agent.result = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "execution_time_ms": 60000,  # 超时时间
                "extracted_answer": "执行超时，无法获取回答",
                "usage_metadata": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_token_details": {},
                    "output_token_details": {}
                }
            }
        except Exception as e:
            print(f"{Colors.RED}警告: 智能体 {agent.name} 执行出错: {str(e)}{Colors.ENDC}")
            agent.score = 0.0
            agent.result = {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "execution_time_ms": 0,
                "extracted_answer": f"执行出错: {str(e)}",
                "usage_metadata": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_token_details": {},
                    "output_token_details": {}
                }
            }

# 注册智能体系统
AgentSystemRegistry.register("evoagent", EvoAgent)

if __name__ == "__main__":
    # 测试EvoAgent
    problem = {
        "problem": "一个正整数，它的平方根是 452，求这个正整数。"
    }
    
    # 使用同步版本的run_agent方法
    evo_agent = EvoAgent()
    result = evo_agent.run_agent(problem)
    print(result)
