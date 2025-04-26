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
    """Represent an LLM agent"""
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
        """generate agent response"""
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
    """Extract the final result from the conversation history"""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.llm = ChatOpenAI(
            model=self.model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def extract(self, all_histories: List[List[Dict[str, str]]], problem: str) -> str:
        """
        Extract the final answer from the conversation history of all agents
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
        """Format all conversation histories"""
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
    """Multi-agent evaluation system based on iterative debate"""
    
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
        """Create multiple agent instances"""
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
        """Generate specific system prompts for each agent"""
        base_prompt = """你是一个专业的问题解决专家，需要：
1. 仔细分析问题和其他专家的观点
2. 提供你的专业见解
3. 必要时质疑或补充其他专家的观点
4. 确保答案准确且符合逻辑"""
        
        return f"{base_prompt}\n你是专家 {agent_index + 1}，专注于提供独特的视角。"

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Run the iterative debate process"""
        problem_text = problem["problem"]
        start_time = time.time()
        
        # Iterative discussion process
        for t in range(self.num_rounds):
            for n, agent in enumerate(self.agents):
                # Generate the response of the current agent
                context = self._build_context(problem_text, n, t)
                response = agent.generate_response(context)
                
                # Add the response to the context of the subsequent agent
                for m in range(n + 1, len(self.agents)):
                    self.agents[m].chat_history.append({
                        "role": "human",
                        "human": f"Expert {n+1}'s response: {response}"
                    })
        
        # Extract the final answer
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
        """Build the context of the current agent"""
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
        """Extract the answer from \boxed{} in the text"""
        import re
        match = re.search(r'\\boxed{(.*?)}', text)
        return match.group(1) if match else text

# Register agents system
AgentSystemRegistry.register(
    "chateval",
    ChatEval,
    num_agents=3,
    num_rounds=2
)

if __name__ == "__main__":
    # text
    problem = {
        "problem": "如果一个正整数是偶数，那么它一定是2的倍数。"
    }
    agent = ChatEval(name="chateval", config={"num_agents": 3, "num_rounds": 2})
    result = agent.run_agent(problem)
    print(result)
