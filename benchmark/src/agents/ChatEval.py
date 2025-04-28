# benchmark/src/agents/ChatEval.py

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

    def generate_response(self, context: str) -> Any:
        """生成代理响应"""
        messages = [
            SystemMessage(content=self.system_prompt),
            *[HumanMessage(content=msg["human"]) if msg.get("role") == "human" 
              else AIMessage(content=msg["ai"]) 
              for msg in self.chat_history],
            HumanMessage(content=context)
        ]
        
        response = self.llm.invoke(messages)
        # 设置AI消息的名字
        response.name = self.name
        
        self.chat_history.append({
            "role": "human",
            "human": context
        })
        self.chat_history.append({
            "role": "ai",
            "ai": response.content
        })
        # 返回完整的响应对象，而不仅仅是内容
        return response

class ResultExtractor:
    """从对话历史中提取最终结果"""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
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
        Original problem: {problem}
        
        Below are the discussion histories of multiple AI agents:
        
        {self._format_histories(all_histories)}
        
        Please analyze the above discussions and provide a final answer. Requirements:
        1. Synthesize all agents' viewpoints
        2. Choose the most reasonable solution
        3. Output the answer in standard mathematical format: \\boxed{{answer}}
        """
        
        messages = [
            SystemMessage(content="You are a professional result analyzer, responsible for extracting the final answer from discussions of multiple AI agents."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

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
        
        # 初始化代理
        self.agents = self._create_agents()
        self.extractor = ResultExtractor(self.model_name)

    def _create_agents(self) -> List[Agent]:
        """创建多个代理实例"""
        agents = []
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for i in range(self.num_agents):
            agent = Agent(
                agent_id=f"agent_{i+1}",
                name=agent_names[i],
                model_name=self.model_name,
                system_prompt=self._get_agent_prompt(i)
            )
            agents.append(agent)
        return agents

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
        all_responses = []
        
        # 迭代讨论过程
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        for t in range(self.num_rounds):
            for n, agent in enumerate(self.agents):
                # 生成当前代理的响应
                context = self._build_context(problem_text, n, t)
                response = agent.generate_response(context)
                
                # 保存响应对象
                all_responses.append(response)
                
                # 将响应添加到后续代理的上下文
                for m in range(n + 1, len(self.agents)):
                    self.agents[m].chat_history.append({
                        "role": "human",
                        "human": f"{agent_names[n]}'s response: {response.content}"
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
            "execution_time_ms": duration_ms,
            "messages": all_responses  # 添加messages字段，包含所有LLM响应对象
        }

    def _build_context(self, problem: str, agent_index: int, round_num: int) -> str:
        """构建当前代理的上下文"""
        agent_names = ["Math Expert", "Logic Expert", "Critical Thinking Expert"]
        agent_name = agent_names[agent_index]
        
        if round_num == 0 and agent_index == 0:
            return f"Please solve this problem: {problem}"
        
        return f"""Round {round_num + 1}, {agent_name}
        
Original problem: {problem}

Please provide your insights based on previous discussions. You can:
1. Agree with and supplement previous viewpoints
2. Propose different solutions
3. Point out potential issues with previous solutions
4. Provide new ideas or methods
5. Do not overly expand to other problems"""

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

if __name__ == "__main__":
    # 测试
    problem = {
        "problem": "一个正整数，它的平方根是 452，求这个正整数。"
    }
    agent = ChatEval(name="chateval", config={"num_agents": 3, "num_rounds": 2})
    result = agent.run_agent(problem)
    print(result)
