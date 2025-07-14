import asyncio

from dotenv import load_dotenv
from mas_arena.agents import AgentSystemRegistry
load_dotenv()
config = {}
single_agent = AgentSystemRegistry.get("single_agent",config)
problem = {"problem":"你是谁"}
result =  asyncio.run(single_agent.run_agent(problem))
print(result)