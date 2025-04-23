import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

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
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def generate_response(self, context: str) -> str:
        """生成代理响应"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history],
            HumanMessage(content=context)
        ]
        
        response = self.llm.invoke(messages)
        self.chat_history.append({
            "role": "human",
            "human": context
        })
        self.chat_history.append({
            "role": "ai",
            "ai": response.content
        })
        return response.content

class ResultExtractor:
    """从对话历史中提取最终结果"""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4")
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> str:
        """
        从所有代理的对话历史中提取最终答案
        """
        # 构建提示
        prompt = f"""
        原始问题：{problem}
        
        以下是多个AI代理的讨论历史：
        
        {self._format_histories(all_histories)}
        
        请分析以上讨论，提供最终答案。要求：
        1. 综合所有代理的观点
        2. 选择最合理的解决方案
        3. 以数学问题的标准格式输出答案：\\boxed{{answer}}
        """
        
        messages = [
            SystemMessage(content="你是一个专业的结果分析器，负责从多个AI代理的讨论中提取最终答案。"),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _format_histories(self, all_histories: List[List[Dict[str, str]]]) -> str:
        """格式化所有对话历史"""
        formatted = []
        for i, history in enumerate(all_histories):
            formatted.append(f"\n代理 {i+1} 的讨论：")
            for msg in history:
                if msg.get("role") == "human":
                    formatted.append(f"问题：{msg['human']}")
                else:
                    formatted.append(f"回答：{msg['ai']}")
        return "\n".join(formatted)

class ChatEval(AgentSystem):
    """基于迭代辩论的多智能体评估系统"""
    
    def __init__(self, name: str = "chateval", config: Dict[str, Any] = None):
        super().__init__(name, config)
        self.config = config or {}
        self.num_agents = self.config.get("num_agents", 3)
        self.num_rounds = self.config.get("num_rounds", 2)
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4")
        
        # 初始化代理
        self.agents = self._create_agents()
        self.extractor = ResultExtractor(self.model_name)

    def _create_agents(self) -> List[Agent]:
        """创建多个代理实例"""
        agents = []
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=f"agent_{i+1}",
                name=f"Expert {i+1}",
                model_name=self.model_name,
                system_prompt=self._get_agent_prompt(i)
            )
            agents.append(agent)
        return agents

    def _get_agent_prompt(self, agent_index: int) -> str:
        """为每个代理生成特定的系统提示"""
        base_prompt = """你是一个专业的问题解决专家，需要：
1. 仔细分析问题和其他专家的观点
2. 提供你的专业见解
3. 必要时质疑或补充其他专家的观点
4. 确保答案准确且符合逻辑"""
        
        return f"{base_prompt}\n你是专家 {agent_index + 1}，专注于提供独特的视角。"

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """运行迭代辩论过程"""
        problem_text = problem["problem"]
        start_time = time.time()
        
        # 迭代讨论过程
        for t in range(self.num_rounds):
            for n, agent in enumerate(self.agents):
                # 生成当前代理的响应
                context = self._build_context(problem_text, n, t)
                response = agent.generate_response(context)
                
                # 将响应添加到后续代理的上下文
                for m in range(n + 1, len(self.agents)):
                    self.agents[m].chat_history.append({
                        "role": "human",
                        "human": f"Expert {n+1}'s response: {response}"
                    })
        
        # 提取最终答案
        all_histories = [agent.chat_history for agent in self.agents]
        final_answer = self.extractor.extract(all_histories, problem_text)
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return {
            "final_answer": final_answer,
            "extracted_answer": self._extract_boxed_answer(final_answer),
            "agent_discussions": all_histories,
            "execution_time_ms": duration_ms
        }

    def _build_context(self, problem: str, agent_index: int, round_num: int) -> str:
        """构建当前代理的上下文"""
        if round_num == 0 and agent_index == 0:
            return f"请解决这个问题：{problem}"
        
        return f"""Round {round_num + 1}, Expert {agent_index + 1}
        
原始问题：{problem}

请基于之前的讨论提供你的见解。你可以：
1. 同意并补充之前的观点
2. 提出不同的解决方案
3. 指出之前解决方案的潜在问题
4. 提供新的思路或方法"""

    def _extract_boxed_answer(self, text: str) -> str:
        """从文本中提取 \boxed{} 中的答案"""
        import re
        match = re.search(r'\\boxed{(.*?)}', text)
        return match.group(1) if match else text

# 注册代理系统
AgentSystemRegistry.register(
    "chateval",
    ChatEval,
    num_agents=3,
    num_rounds=2
)
