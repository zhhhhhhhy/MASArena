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

# 为结构化输出定义TypedDict类
class ExpertTeam(TypedDict):
    """专家团队配置"""
    agents: List[Dict[str, Any]]

class ExpertSolution(TypedDict):
    """专家解决方案"""
    analysis: str
    solution: str
    confidence: int  # 1-5分表示专家对自己解答的信心

class EvaluationResult(TypedDict):
    """评估结果"""
    status: str  # "complete" 或 "need_new_experts"
    final_solution: str  # 最终解决方案
    feedback: str  # 反馈意见（如果需要新专家）
    reasoning: str  # 评估理由
    improvement_score: float  # 与上次迭代相比的改进程度（0-1）
    solution_quality: float  # 解决方案质量评分（0-1）

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
            3. Each expert should have specialized knowledge relevant to the problem
            4. Each expert should have a clearly defined role with specific responsibilities
            5. The team should collectively cover all aspects of the problem"""
        )
        # 使用结构化输出初始化LLM
        self.llm = ChatOpenAI(
            model=self.model_name
        )
        
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

            Please analyze the problem carefully and identify what specialized knowledge would be needed to solve it.
            Then create a team of experts with complementary skills that together can address all aspects of the problem.

            For each expert, provide:
            1. A descriptive name reflecting their expertise area
            2. A detailed description of their role and responsibilities
            3. An ID number (starting from 1)

            Think step by step about different aspects of the problem and how each expert will contribute.
            If feedback was provided, make sure your new team addresses those specific concerns.

            Agent ID: {self.agent_id}
        """

    def describe(self, problem: str, feedback: str = None):
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._create_prompt(problem, feedback))
        ]

        start_time = time.time()
        end_time = start_time  # 初始化end_time，防止异常情况下未定义
        
        try:
            # 使用结构化输出调用LLM
            llm_with_schema = self.llm.with_structured_output(schema=ExpertTeam, include_raw=True)
            response = llm_with_schema.invoke(messages)
            end_time = time.time()  # 更新end_time
            
            # 从结构化响应中提取内容
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # 验证结构化数据
            if not isinstance(structured_data, dict) or "agents" not in structured_data or not structured_data["agents"]:
                print(f"Warning: Invalid or empty response from recruiter. Raw content: {raw_response.content[:200]}...")
                # 从原始响应尝试提取专家信息
                try:
                    # 尝试解析JSON
                    import re
                    # 查找可能的JSON对象
                    json_match = re.search(r'(\{.*\})', raw_response.content.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(1)
                        parsed_data = json.loads(potential_json)
                        if "agents" in parsed_data and parsed_data["agents"]:
                            structured_data = parsed_data
                        else:
                            # 创建默认专家团队
                            structured_data = {"agents": self._create_default_experts()}
                    else:
                        structured_data = {"agents": self._create_default_experts()}
                except Exception as parse_err:
                    print(f"Error parsing recruiter response: {str(parse_err)}")
                    structured_data = {"agents": self._create_default_experts()}
            
            
            # 设置名称
            raw_response.name = f"recruiter_{self.agent_id}"
            
            return {
                "agent_id": self.agent_id,
                "solution": structured_data,
                "message": raw_response,  # 保存原始消息以保留usage_metadata
                "latency_ms": (end_time - start_time) * 1000,
            }
            
        except Exception as e:
            # 如果结构化输出失败，回退到标准模式
            print(f"Structured output failed for recruiter: {str(e)}. Falling back to standard output.")
            
            # 重新调用模型，不使用结构化输出
            response = self.llm.invoke(messages)
            end_time = time.time()
            
            # 设置名称
            response.name = f"recruiter_{self.agent_id}"
            
            # 尝试从响应内容中提取JSON
            try:
                # 尝试直接解析为JSON
                content_text = response.content
                try:
                    content_json = json.loads(content_text)
                    if "agents" in content_json and content_json["agents"]:
                        structured_data = content_json
                    else:
                        structured_data = {"agents": self._create_default_experts()}
                except json.JSONDecodeError:
                    # 尝试在文本中查找JSON部分
                    import re
                    json_match = re.search(r'(\{.*\})', content_text.replace('\n', ' '), re.DOTALL)
                    if json_match:
                        potential_json = json_match.group(1)
                        try:
                            parsed_json = json.loads(potential_json)
                            if "agents" in parsed_json and parsed_json["agents"]:
                                structured_data = parsed_json
                            else:
                                structured_data = {"agents": self._create_default_experts()}
                        except Exception as e:
                            print(f"[WARNING] Error parsing recruiter content: {str(e)}")
                            structured_data = {"agents": self._create_default_experts()}
                    else:
                        # 无法找到有效的JSON，创建默认专家
                        structured_data = {"agents": self._create_default_experts()}
            except Exception as parse_error:
                print(f"[WARNING] Error parsing recruiter content: {str(parse_error)}")
                structured_data = {"agents": self._create_default_experts()}
            
            return {
                "agent_id": self.agent_id,
                "solution": structured_data,
                "message": response,
                "latency_ms": (end_time - start_time) * 1000,
            }
    
    def _create_default_experts(self) -> List[Dict[str, Any]]:
        """创建默认专家团队，当结构化输出失败时使用"""
        default_experts = []
        expert_types = [
            {"name": "数学专家", "describe": "专门处理数学问题、计算和证明的专家。"},
            {"name": "问题分析专家", "describe": "负责分析问题结构、拆解复杂问题的专家。"},
            {"name": "解决方案专家", "describe": "整合分析结果，提供完整解决方案的专家。"}
        ]
        
        # 根据设置的数量创建专家
        for i in range(1, min(self.num_agents + 1, len(expert_types) + 1)):
            expert = expert_types[i-1].copy()
            expert["agent_id"] = i
            default_experts.append(expert)
        
        return default_experts

class WorkAgent:
    """Work agent that solves specific aspects of a problem"""
    def __init__(self, agent_id: str, system_prompt: str = None):
        self.agent_id = agent_id
        self.model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.system_prompt = (
            f"{system_prompt}\n"
            "## Output Requirements:\n"
            "1. Analyze the problem from your expert perspective\n"
            "2. Provide a detailed solution for your specific part of the problem\n"
            "3. Rate your confidence in your solution (1-5 scale, with 5 being highest)\n"
            "4. Structure your response logically with clear reasoning"
        )
        self.llm = ChatOpenAI(
            model=self.model_name,
            max_tokens=1000
        )

    def solve(self, problem: str, feedback: str = None):
        """Solve a problem with optional feedback"""
        problem_content = problem
        if feedback:
            problem_content += f"\n\nFeedback from previous evaluation:\n{feedback}\nPlease address this feedback in your solution."
            
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=problem_content)
        ]

        start_time = time.time()
        end_time = start_time  # 初始化end_time，防止异常情况下未定义
        
        try:
            # 尝试使用结构化输出
            llm_with_schema = self.llm.with_structured_output(schema=ExpertSolution, include_raw=True)
            response = llm_with_schema.invoke(messages)
            end_time = time.time()  # 更新end_time
            
            # 从结构化响应中提取内容
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # 确保structured_data中包含所需的所有字段
            if "solution" not in structured_data:
                structured_data["solution"] = raw_response.content
            if "analysis" not in structured_data:
                structured_data["analysis"] = "No analysis provided"
            if "confidence" not in structured_data:
                structured_data["confidence"] = 3  # 默认中等信心
            
            
            # 设置名称
            raw_response.name = f"expert_{self.agent_id}"
            
            return {
                "agent_id": self.agent_id,
                "solution": structured_data["solution"],  # 为了兼容现有代码，直接提取solution字段
                "structured_solution": structured_data,  # 保存完整的结构化数据
                "message": raw_response,  # 保存原始消息以保留usage_metadata
                "latency_ms": (end_time - start_time) * 1000,
            }
            
        except Exception as e:
            # 如果结构化输出失败，回退到标准模式
            print(f"Structured output failed for agent {self.agent_id}: {str(e)}. Falling back to standard output.")
            
            # 重新调用模型，不使用结构化输出
            end_time = time.time()  # 记录之前尝试的时间
            start_time = time.time()  # 重新计时
            
            response = self.llm.invoke(messages)
            end_time = time.time()
            
            # 设置名称
            response.name = f"expert_{self.agent_id}"
            
            return {
                "agent_id": self.agent_id,
                "solution": response.content,
                "message": response,
                "latency_ms": (end_time - start_time) * 1000,
            }

class Evaluator:
    """Evaluates agent solutions and decides whether to recruit new experts or provide final solution"""
    def __init__(self, model_name: str = None, max_iterations: int = 3, min_quality_threshold: float = 0.7, min_improvement_threshold: float = 0.1):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.max_iterations = max_iterations
        self.min_quality_threshold = min_quality_threshold  # 最低解决方案质量阈值
        self.min_improvement_threshold = min_improvement_threshold  # 最低改进阈值
        self.previous_solution_quality = 0  # 上一轮解决方案的质量
        self.llm = ChatOpenAI(
            model=self.model_name
        )
        
    def evaluate(self, problem: str, solutions: List[Dict[str, Any]], iteration: int, previous_solutions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate solutions from multiple agents and decide whether to:
        1. Provide final solution if satisfactory
        2. Provide feedback for another round of recruitment
        
        Args:
            problem: Original problem description
            solutions: List of solutions from agents
            iteration: Current iteration count
            previous_solutions: Solutions from previous iteration for comparison
            
        Returns:
            Dictionary with evaluation results
        """
        # 提取结构化解决方案（如果可用）
        solutions_details = []
        for sol in solutions:
            try:
                if "structured_solution" in sol:
                    # 使用结构化解决方案
                    structured = sol["structured_solution"]
                    solutions_details.append(
                        f"Expert {sol['agent_id']}:\n"
                        f"Analysis: {structured.get('analysis', 'No analysis provided')}\n"
                        f"Solution: {structured.get('solution', 'No solution provided')}\n"
                        f"Confidence: {structured.get('confidence', 3)}/5\n"
                    )
                else:
                    # 使用普通解决方案
                    solutions_details.append(f"Expert {sol['agent_id']} solution:\n{sol.get('solution', 'No solution provided')}\n")
            except Exception as e:
                print(f"Error processing solution from agent {sol.get('agent_id', 'unknown')}: {str(e)}")
                solutions_details.append(f"Expert {sol.get('agent_id', 'unknown')} solution:\nError: Could not process solution\n")
                
        solutions_text = "\n\n".join(solutions_details)
        
        # 如果有上一轮的解决方案，添加到提示中进行比较
        previous_solutions_text = ""
        if previous_solutions and len(previous_solutions) > 0:
            prev_details = []
            for sol in previous_solutions:
                try:
                    if "structured_solution" in sol:
                        structured = sol["structured_solution"]
                        prev_details.append(
                            f"Expert {sol['agent_id']}:\n"
                            f"Analysis: {structured.get('analysis', 'No analysis provided')}\n"
                            f"Solution: {structured.get('solution', 'No solution provided')}\n"
                            f"Confidence: {structured.get('confidence', 3)}/5\n"
                        )
                    else:
                        prev_details.append(f"Expert {sol['agent_id']} solution:\n{sol.get('solution', 'No solution provided')}\n")
                except Exception:
                    continue
            
            if prev_details:
                previous_solutions_text = "\n\nPrevious iteration solutions:\n" + "\n\n".join(prev_details)
        
        prompt = f"""
        I need you to analyze multiple expert solutions to the same problem.
        
        Original problem:
        {problem}
        
        Current expert solutions:
        {solutions_text}
        {previous_solutions_text}
        
        This is iteration {iteration} out of {self.max_iterations}.
        
        Your task:
        1. Analyze each expert's solution and their confidence level
        2. Determine if the solutions collectively solve the problem satisfactorily
        3. If solutions are adequate, compile them into a comprehensive final solution
        4. If solutions need improvement, provide specific feedback for recruiting better experts
        5. Rate the overall quality of the current solutions (0-1 scale, with 1 being perfect)
        6. If there are previous solutions, rate the improvement from previous to current solutions (0-1 scale)
        
        If you decide the solutions collectively solve the problem:
        - Provide a detailed final solution combining the best insights from all experts
        - Include step-by-step reasoning
        - {self.format_prompt} 
        
        If you decide the solutions need improvement:
        - Explain what aspects of the problem remain inadequately addressed
        - Provide specific feedback on what expertise is missing or needs enhancement
        
        In addition to deciding whether to continue or stop, you must provide two numerical scores:
        1. solution_quality (0-1): How good is the current solution? (1 = perfect solution, 0 = no progress)
        2. improvement_score (0-1): How much improvement compared to previous iteration? (1 = major improvement, 0 = no improvement)
        """
        
        messages = [
            SystemMessage(content="You are an expert evaluator that analyzes multiple solutions and determines if they adequately solve the problem. You can either provide a final comprehensive solution or request improvements with specific feedback."),
            HumanMessage(content=prompt)
        ]
        
        start_time = time.time()
        end_time = start_time  # 初始化end_time，防止异常情况下未定义
        
        try:
            # 使用结构化输出调用LLM
            llm_with_schema = self.llm.with_structured_output(schema=EvaluationResult, include_raw=True)
            response = llm_with_schema.invoke(messages)
            end_time = time.time()  # 更新end_time
            
            # 从结构化响应中提取内容
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # 确保structured_data中包含所需的所有字段
            if "status" not in structured_data:
                structured_data["status"] = "need_new_experts" if iteration < self.max_iterations else "complete"
            if "final_solution" not in structured_data:
                structured_data["final_solution"] = raw_response.content
            if "feedback" not in structured_data:
                structured_data["feedback"] = ""
            if "reasoning" not in structured_data:
                structured_data["reasoning"] = "No reasoning provided"
            if "solution_quality" not in structured_data:
                structured_data["solution_quality"] = 0.5  # 默认中等质量
            if "improvement_score" not in structured_data:
                structured_data["improvement_score"] = 0.1  # 默认微小改进
                
            # 智能终止决策逻辑
            current_quality = structured_data["solution_quality"]
            improvement = structured_data["improvement_score"]
            
            # 条件1: 如果质量已经超过阈值，可以提前完成
            if current_quality >= self.min_quality_threshold:
                structured_data["status"] = "complete"
                print(f"Solution quality {current_quality} exceeds threshold {self.min_quality_threshold}, completing early.")
            
            # 条件2: 如果改进低于阈值且不是首次迭代，可能陷入停滞
            if iteration > 1 and improvement < self.min_improvement_threshold:
                structured_data["status"] = "complete"
                print(f"Improvement {improvement} below threshold {self.min_improvement_threshold}, stopping iterations.")
            
            # 条件3: 如果已达到最大迭代次数，必须完成
            if iteration >= self.max_iterations:
                structured_data["status"] = "complete"
                print(f"Reached maximum iterations ({self.max_iterations}), completing.")
            
            # 存储当前质量评分以供下次迭代比较
            self.previous_solution_quality = current_quality
            
            
            # 设置名称
            raw_response.name = "evaluator"
            
            return {
                "final_solution": structured_data["final_solution"],
                "message": raw_response,  # 保存原始消息以保留usage_metadata
                "latency_ms": (end_time - start_time) * 1000,
                "evaluation": structured_data,
            }
            
        except Exception as e:
            # 如果结构化输出失败，回退到标准模式并尝试解析JSON
            print(f"Structured output failed for evaluator: {str(e)}. Falling back to standard output and JSON parsing.")
            
            # 重新调用模型，不使用结构化输出
            response = self.llm.invoke(messages)
            end_time = time.time()
            
            # 设置名称
            response.name = "evaluator"
            
            # 尝试从响应中解析JSON
            try:
                # 清理响应，移除markdown代码块
                content = response.content
                import re
                content = re.sub(r'```(?:json)?', '', content)
                content = content.strip()
                content = re.sub(r'```$', '', content).strip()
                
                # 尝试直接解析JSON
                try:
                    evaluation = json.loads(content)
                except json.JSONDecodeError:
                    # 如果解析失败，尝试使用正则表达式提取JSON对象
                    json_match = re.search(r'({.*})', content.replace('\n', ' '), re.DOTALL)
                    
                    if json_match:
                        json_str = json_match.group(1)
                        # 处理转义序列
                        json_str = json_str.replace('\\', '\\\\')
                        evaluation = json.loads(json_str)
                    else:
                        # 如果仍然无法提取，创建默认评估
                        evaluation = {
                            "status": "need_new_experts" if iteration < self.max_iterations else "complete",
                            "final_solution": response.content,
                            "feedback": f"Could not extract structured evaluation. Please provide a team that can solve: {problem[:100]}...",
                            "reasoning": "Error parsing evaluation",
                            "solution_quality": 0.5,
                            "improvement_score": 0.1
                        }
            except Exception as parse_error:
                print(f"Error parsing evaluator response: {str(parse_error)}")
                # 创建默认评估
                evaluation = {
                    "status": "need_new_experts" if iteration < self.max_iterations else "complete",
                    "final_solution": response.content,
                    "feedback": f"Error processing evaluation. Please provide a team that can solve: {problem[:100]}...",
                    "reasoning": "Error in evaluation process",
                    "solution_quality": 0.5,
                    "improvement_score": 0.1
                }
            
            # 确保所有必须的字段都存在
            if "status" not in evaluation:
                evaluation["status"] = "need_new_experts" if iteration < self.max_iterations else "complete"
            if "final_solution" not in evaluation:
                evaluation["final_solution"] = response.content
            if "feedback" not in evaluation:
                evaluation["feedback"] = ""
            if "reasoning" not in evaluation:
                evaluation["reasoning"] = "No reasoning provided"
            if "solution_quality" not in evaluation:
                evaluation["solution_quality"] = 0.5
            if "improvement_score" not in evaluation:
                evaluation["improvement_score"] = 0.1
            
            # 智能终止决策逻辑
            current_quality = evaluation["solution_quality"]
            improvement = evaluation["improvement_score"]
            
            # 条件1: 如果质量已经超过阈值，可以提前完成
            if current_quality >= self.min_quality_threshold:
                evaluation["status"] = "complete"
                print(f"Solution quality {current_quality} exceeds threshold {self.min_quality_threshold}, completing early.")
            
            # 条件2: 如果改进低于阈值且不是首次迭代，可能陷入停滞
            if iteration > 1 and improvement < self.min_improvement_threshold:
                evaluation["status"] = "complete"
                print(f"Improvement {improvement} below threshold {self.min_improvement_threshold}, stopping iterations.")
            
            # 条件3: 如果已达到最大迭代次数，必须完成
            if iteration >= self.max_iterations:
                evaluation["status"] = "complete"
                print(f"Reached maximum iterations ({self.max_iterations}), completing.")
            
            # 存储当前质量评分以供下次迭代比较
            self.previous_solution_quality = current_quality
                
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
        
        # 新增的质量控制与早停配置
        self.min_quality_threshold = self.config.get("min_quality_threshold", 0.7)
        self.min_improvement_threshold = self.config.get("min_improvement_threshold", 0.1)
        self.early_stopping_rounds = self.config.get("early_stopping_rounds", 2)
        self.max_runtime = self.config.get("max_runtime", 300)  # 默认最大运行时间5分钟
        
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
        
        # 从结构化输出获取专家配置
        expert_config = response_dict.get("solution", {})
        agents_list = expert_config.get("agents", [])
        
        # 创建专家团队
        expert_team = []
        for idx, agent in enumerate(agents_list, 1):
            # 确保字典中有必要的字段
            if isinstance(agent, dict):
                agent_id = agent.get("agent_id", str(idx))
                if not isinstance(agent_id, str):
                    agent_id = str(agent_id)
                    
                expert_team.append(
                    ExpertProfile(
                        id=agent_id,
                        name=agent.get("name", f"Expert {agent_id}"),
                        description=agent.get("describe", agent.get("description", ""))[:500]  # 支持不同的字段名并截断长描述
                    )
                )
        
        # 如果没有获取到专家，创建默认专家
        if not expert_team:
            print("Warning: No experts found in recruiter response, creating default experts")
            for i in range(1, self.num_agents + 1):
                expert_team.append(
                    ExpertProfile(
                        id=str(i),
                        name=f"General Expert {i}",
                        description="A general expert who can solve various aspects of the problem."
                    )
                )
        
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
        
        # Initialize messages and solutions
        all_messages = []
        all_solutions = []
        feedback = None
        final_solution = None
        
        # 跟踪上一轮的解决方案，用于比较改进
        previous_solutions = None
        
        # 使用类属性而非从config获取
        min_quality = self.min_quality_threshold
        min_improvement = self.min_improvement_threshold
        early_stopping_rounds = self.early_stopping_rounds
        max_runtime = self.max_runtime
        
        # 跟踪连续几轮没有明显改进
        no_improvement_count = 0
        start_runtime = time.time()
        
        # Create evaluator
        evaluator = Evaluator(
            model_name=self.model_name, 
            max_iterations=self.max_iterations,
            min_quality_threshold=min_quality,
            min_improvement_threshold=min_improvement
        )
        
        # Run iterations until evaluator is satisfied or max iterations reached
        for iteration in range(1, self.max_iterations + 1):
            # 检查是否超过最大运行时间
            current_runtime = time.time() - start_runtime
            if current_runtime > max_runtime:
                print(f"Reached maximum runtime ({max_runtime}s), stopping at iteration {iteration}")
                break
            
            print(f"Starting iteration {iteration}/{self.max_iterations}")
            
            # Create specialized agents for this problem with feedback from previous iteration
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
                    
                # Run all agents asynchronously
                agent_solutions = loop.run_until_complete(
                    self._async_solve_problem(problem_text, agents)
                )
            else:
                # Run agents sequentially
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
            
            # 获取上一轮的专家解决方案（如果有）
            if iteration > 1 and len(all_solutions) > 1:
                previous_solutions = all_solutions[iteration-2]["solutions"]
            
            # Evaluate solutions，传递上一轮的解决方案用于比较
            evaluation_result = evaluator.evaluate(problem_text, agent_solutions, iteration, previous_solutions)
            evaluation = evaluation_result.get("evaluation", {})
            
            # Add evaluator message
            if "message" in evaluation_result:
                all_messages.append(evaluation_result["message"])
            
            # 获取改进分数，如果低于阈值则增加无改进计数
            improvement_score = evaluation.get("improvement_score", 0)
            if iteration > 1 and improvement_score < min_improvement:
                no_improvement_count += 1
            else:
                no_improvement_count = 0  # 重置计数
            
            # 检查是否达到早停条件
            if no_improvement_count >= early_stopping_rounds:
                print(f"Early stopping after {no_improvement_count} rounds with no significant improvement")
                # 使用当前最佳解决方案
                final_solution = evaluation.get("final_solution", "")
                break
            
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
            last_evaluation = evaluator.evaluate(problem_text, all_solutions[-1]["solutions"], self.max_iterations, all_solutions[:-1] if len(all_solutions) > 1 else None)
            final_solution = last_evaluation.get("evaluation", {}).get("final_solution", "No satisfactory solution found")
            # Add final evaluator message
            if "message" in last_evaluation:
                all_messages.append(last_evaluation["message"])
        
        # For math problems, ensure the final solution is properly formatted
        if isinstance(final_solution, (int, float)):
            final_solution = f"The answer is \\boxed{{{final_solution}}}"
        
        # Filter messages to only include those with usage_metadata for evaluation framework
        messages_with_metadata = [msg for msg in all_messages if hasattr(msg, 'usage_metadata') and msg.usage_metadata]
        
        # Return final answer and all messages
        return {
            "messages": messages_with_metadata,  # 只返回带有usage_metadata的消息
            "final_answer": final_solution,
            "agent_solutions": all_solutions,
        }

# Register the agent system with default parameters
# 确保这些默认值与AgentVerse类中的默认值保持一致
AgentSystemRegistry.register("agentverse", AgentVerse, num_agents=3, parallel=True, max_iterations=3)

if __name__ == "__main__":
    problem = {
        "problem": "What is the sum of the first 100 natural numbers?",
        "id": "problem_1"
    }
    result = AgentVerse().run_agent(problem)
    print(result)
