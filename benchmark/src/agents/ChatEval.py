# benchmark/src/agents/ChatEval.py

import time
import json
import os
from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

# 定义结构化输出类，使用TypedDict代替Pydantic
class AgentResponse(TypedDict):
    """智能体响应的结构化输出"""
    analysis: str  # 问题分析
    solution: str  # 解决方案
    confidence: int  # 解答信心程度，范围1-5

@dataclass
class Agent:
    """代表一个LLM代理"""
    agent_id: str
    name: str
    model_name: str
    system_prompt: str
    chat_history: List[Dict[str, str]] = None
    
    def __post_init__(self):
        self.chat_history = []
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,  # 设置请求超时为60秒
            max_retries=2        # 设置最大重试次数为2
        )

    def generate_response(self, context: str) -> Any:
        """生成代理响应"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history],
            HumanMessage(content=context)
        ]
        
        # 使用结构化输出
        try:
            llm_with_schema = self.llm.with_structured_output(schema=AgentResponse, include_raw=True)
            response = llm_with_schema.invoke(messages)
            
            # 获取结构化数据和原始响应
            structured_data = response["parsed"]
            raw_response = response["raw"]
            
            # 确保structured_data是字典而不是对象
            if hasattr(structured_data, "dict"):
                structured_data = structured_data.dict()
            elif hasattr(structured_data, "model_dump"):
                structured_data = structured_data.model_dump()
            
            # 设置AI消息的名字
            raw_response.name = self.name
            
            # 更新聊天历史
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": raw_response.content
            })
            
            # 返回原始响应对象，保留usage_metadata
            return {
                "message": raw_response,
                "structured_solution": structured_data,
                "solution": raw_response.content
            }
            
        except Exception as e:
            print(f"结构化输出失败: {str(e)}，回退到标准输出")
            
            # 回退到标准输出
            response = self.llm.invoke(messages)
            response.name = self.name
            
            self.chat_history.append({
                "role": "human",
                "human": context
            })
            self.chat_history.append({
                "role": "ai",
                "ai": response.content
            })
            
            return {
                "message": response,
                "solution": response.content
            }

class ResultExtractor:
    """从对话历史中提取最终结果"""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=self.model_name,
            request_timeout=60,  # 设置请求超时为60秒
            max_retries=2        # 设置最大重试次数为2
        )
        self.name = "result_extractor"
        
    def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> Dict[str, Any]:
        """
        从所有代理的对话历史中提取最终答案
        """
        # 根据问题类型选择不同的提示
        prompt = f"""Original problem: {problem}

Below are the discussion histories of multiple AI agents:

{self._format_histories(all_histories)}

Please analyze the above discussions and provide a final answer. Requirements:
- Synthesize all agents' viewpoints.
- Choose the most reasonable solution/option.
{self.format_promt}
"""
  
        messages = [
            SystemMessage(content="You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            response.name = "evaluator"
            return {
                "message": response
            }
        except Exception as e:
            print(f"调用 LLM 失败: {str(e)}")
            return {
                "message": None
            }

    def _format_histories(self, all_histories: List[List[Dict[str, str]]]) -> str:
        """格式化所有对话历史"""
        formatted = []
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for i, history in enumerate(all_histories):
            formatted.append(f"\n{agent_names[i]}'s discussion:")
            for msg in history:
                if msg.get("role") == "human":
                    formatted.append(f"Question: {msg['human']}")
                else:
                    formatted.append(f"Answer: {msg['ai']}")
        return "\n".join(formatted)
        

class ChatEval(AgentSystem):
    """基于迭代辩论的多智能体评估系统"""
    
    def __init__(self, name: str = "chateval", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.num_rounds = self.config.get("num_rounds", 2)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o-mini")
        
        # 初始化代理 and extractor via _create_agents
        # self.agents and self.extractor will be set by _create_agents
        agent_components = self._create_agents()
        self.agents = [w for w in agent_components["workers"] if isinstance(w, Agent)]
        extractors = [w for w in agent_components["workers"] if isinstance(w, ResultExtractor)]
        if not extractors:
            raise ValueError("ResultExtractor not found in components created by _create_agents.")
        self.extractor = extractors[0]

    def _create_agents(self) -> List[Agent]:
        """创建多个代理实例和结果提取器"""
        # This method will be patched by ToolIntegrationWrapper if this system is wrapped.
        # The wrapper expects a dictionary: {"workers": [worker1, worker2, ...]}
        # Each worker should have a .name and .llm attribute.
        
        debate_agents = []
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=f"agent_{i+1}",
                name=agent_names[i],
                model_name=self.model_name,
                system_prompt=self._get_agent_prompt(i)
            )
            debate_agents.append(agent)
        
        # Create and assign the extractor here
        extractor = ResultExtractor(self.model_name)
        # self.extractor = extractor # Assign to self if needed elsewhere before run_agent completes,
                                 # but __init__ already handles setting self.extractor.

        return {
            "workers": debate_agents + [extractor]
        }

    def _get_agent_prompt(self, agent_index: int) -> str:
        """为每个代理生成特定的系统提示"""
        # 为三个不同角色设置不同的prompt
        if agent_index == 0:
            return """You are a Mathematics Expert, focused on solving mathematical problems. You need to:
1. Carefully analyze the key points of mathematical problems
2. Provide clear mathematical reasoning processes
3. Question or supplement other experts' viewpoints when necessary
4. Ensure answers are accurate and logically sound
5. Use mathematical symbols and formulas to express your thoughts

You are the Mathematics Expert, focused on providing mathematical perspective analysis."""
        elif agent_index == 1:
            return """You are a Logic Expert, focused on logical analysis of problems. You need to:
1. Carefully analyze the logical structure of problems
2. Provide clear reasoning chains
3. Question or supplement other experts' viewpoints when necessary
4. Ensure reasoning processes are rigorous and without loopholes
5. Pay attention to implicit conditions and boundary cases

You are the Logic Expert, focused on providing logical perspective analysis."""
        else:  # agent_index == 2
            return """You are a Critical Thinking Expert, focused on multi-angle analysis of problems. You need to:
1. Carefully analyze multiple aspects of problems
2. Provide comprehensive thinking perspectives
3. Question or supplement other experts' viewpoints when necessary
4. Ensure consideration of various possibilities
5. Pay attention to potential traps and misconceptions

You are the Critical Thinking Expert, focused on providing multi-angle perspective analysis."""

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """运行迭代辩论过程"""
        problem_text = problem["problem"]
        start_time = time.time()

        # 存储所有LLM响应对象
        all_messages = []
        agent_histories = []
        
        # 迭代讨论过程
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for t in range(self.num_rounds):
            for n, agent in enumerate(self.agents):
                # 生成当前代理的响应
                context = self._build_context(problem_text, n, t)
                response_data = agent.generate_response(context)
                
                # 保存响应对象
                if "message" in response_data:
                    all_messages.append(response_data["message"])
                
                # 将响应添加到后续代理的上下文
                solution_text = response_data.get("solution", "")
                for m in range(n + 1, len(self.agents)):
                    self.agents[m].chat_history.append({
                        "role": "human",
                        "human": f"{agent_names[n]}'s response: {solution_text}"
                    })
        
        # 提取所有代理的聊天历史
        agent_histories = [agent.chat_history for agent in self.agents]
        
        # 提取最终答案
        extractor_result = self.extractor.extract(agent_histories, problem_text, options)
        
        # 添加评估器消息
        if "message" in extractor_result and extractor_result["message"]:
            all_messages.append(extractor_result["message"])
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return {
            "messages": all_messages,  # 包含所有LLM响应对象
            "execution_time_ms": duration_ms
        }

    def _build_context(self, problem: str, agent_index: int, round_num: int) -> str:
        """构建当前代理的上下文"""
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        agent_name = agent_names[agent_index]
        
        problem_statement = f"Original problem: {problem}"
        problem_statement += self.format_prompt

        if round_num == 0 and agent_index == 0:
            return f"Please solve this problem or select the best option based on your expertise:\n{problem_statement}"
        
        return f"""Round {round_num + 1}, {agent_name}
        
{problem_statement}

Please provide your insights based on previous discussions. You can:
1. Agree with and supplement previous viewpoints
2. Propose different solutions or select a different option if applicable
3. Point out potential issues with previous solutions/selected options
4. Provide new ideas or methods
5. Do not overly expand to other problems
If the problem is multiple choice, please indicate your chosen option clearly in your response."""

# 注册代理系统
AgentSystemRegistry.register(
    "chateval",
    ChatEval,
    num_agents=3,
    num_rounds=2
)

if __name__ == "__main__":
    # 测试
    problem = {
        "problem": "一个正整数，它的平方根是 452，求这个正整数。"
    }
    agent = ChatEval(name="chateval", config={"num_agents": 3, "num_rounds": 2})
    result = agent.run_agent(problem)
    print(result)
