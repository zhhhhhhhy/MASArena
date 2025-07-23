import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
from dotenv import load_dotenv
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

load_dotenv()

api_key = "sk-3WoJwFaztEeXC3gdkzxwc4c3RKwcpVjtwsJs6YmKzfDuMG6j"
base_url="https://zjuapi.com/v1"

def termination_condition(messages):
    if messages:
        last_message = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        return "APPROVE" in last_message.upper()
    return False
def convert_to_ai_message(self,messages):
    allmessage = []
    for msg in messages:
        ai_message = {
            'content': msg.content,  # 使用点号访问属性
            'name': msg.source,      # 使用点号访问属性
            'role': 'assistant',
            'message_type': 'ai_response',
            'usage_metadata': msg.models_usage  # 使用点号访问属性
        }
        allmessage.append(ai_message)
    return allmessage

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o",api_key=api_key,base_url=base_url)
    primary = AssistantAgent(
        name="primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant, skilled at generating creative and accurate content."
    )

    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message="Provide constructive feedback on the content provided. Respond with 'APPROVE' when the content meets high standards or your feedback has been addressed."
    )
    # system_message="Provide constructive feedback on the content provided. Respond with 'APPROVE' when the content meets high standards or your feedback has been addressed."

    group_chat = RoundRobinGroupChat(
        participants=[primary, critic],
        termination_condition=TextMentionTermination(text="APPROVE"),
        max_turns=3
    )

    question = "什么是人工智能？"

    result = await group_chat.run(task=question)
    allmessage = convert_to_ai_message(result.messages)
    primary_messages = []

    for msg in allmessage:
        if msg['name'] == 'primary':
            primary_messages.append(msg)

    
    final_answer = primary_messages[-1]['content']
    print("final_answer:",final_answer)
    await model_client.close()

asyncio.run(main())
