
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from mas_arena.agents.base import AgentSystem, AgentSystemRegistry

load_dotenv()

class Demo(AgentSystem):
    def __init__(self, name: str = "demo", config: Dict[str, Any] = None):
        """Initialize the AutoGen System"""
        super().__init__(name, config)
        self.config = config or {}
        
        # Default model and agent configurations
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "gpt-4o")
        self.num_rounds = self.config.get("num_rounds", 5)
        
        # LLM configuration
        llm_config = {
            "config_list": [{
                "model": self.model_name,
                "api_key": os.getenv("OPENAI_API_KEY"),
                "base_url": os.getenv("OPENAI_API_BASE")
            }]
        }
        
        # Define AutoGen agents with distinct roles
        self.agents = [
            {
                "name": "primary",
                "agent": AssistantAgent(
                    name="primary",
                    system_message="You are a helpful AI assistant, skilled at generating creative and accurate content.",
                    llm_config=llm_config
                )
            },
            {
                "name": "critic",
                "agent": AssistantAgent(
                    name="critic",
                    system_message="Provide constructive feedback on the content provided. Respond with 'APPROVE' when the content meets high standards or your feedback has been addressed.",
                    llm_config=llm_config
                )
            }
        ]
        
        # User proxy to initiate the chat
        self.user_proxy = UserProxyAgent(
            name="user_proxy",
            code_execution_config=False
        )

    async def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        problem_text = problem["problem"]
        messages = [
            {"role": "user", "content": f"Problem: {problem_text}"}
        ]
        conversation_history = messages.copy()
        all_messages = []
        final_answer = ""
        
        
        for _ in range(self.num_rounds):
            for agent_info in self.agents:
                agent_name = agent_info["name"]
                agent = agent_info["agent"]
                
                # Prepare messages for the current agent
                agent_messages = [
                    {"role": "system", "content": agent.system_message},
                    *conversation_history
                ]
                
                # Initiate chat with the current agent
                chat_result = await self.user_proxy.initiate_chat(
                    recipient=agent,
                    message=agent_messages[-1]["content"],  # Use the latest message content
                    max_turns=1,
                    clear_history=True
                )
                
                # Process chat results
                for message in chat_result.chat_history:
                    if message.get("role") == "assistant":
                        response_content = message.get("content", "")
                        ai_message = {
                            'content': response_content,
                            'name': agent_name,
                            'role': 'assistant',
                            'message_type': 'ai_response',
                            'usage_metadata': getattr(chat_result, 'usage', None)
                        }
                        conversation_history.append({
                            "role": "assistant",
                            "content": response_content,
                            "name": agent_name
                        })
                        all_messages.append(ai_message)
                        
                        if agent_name == "primary":
                            final_answer = response_content
                        
                        # Check for critic approval
                        if agent_name == "critic" and "approve" in response_content.lower():
                            print("Messages:", all_messages)
                            print("Final Answer:", final_answer)
                            return {
                                "messages": all_messages,
                                "final_answer": final_answer
                            }
                    

        
        print("Messages:", all_messages)
        print("Final Answer:", final_answer)
        return {
            "messages": all_messages,
            "final_answer": final_answer
        }

AgentSystemRegistry.register("demo", Demo)

