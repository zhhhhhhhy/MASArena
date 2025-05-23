"""
Multi-Agent Systems for Benchmarking

This package provides various agent system implementations for benchmarking.
"""

from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry, create_agent_system
# from benchmark.src.agents.supervisor_mas import SupervisorMAS
# from benchmark.src.agents.swarm import SwarmSystem
# from benchmark.src.agents.single_agent import SingleAgent
# from benchmark.src.agents.AgentVerse import AgentVerse
# from benchmark.src.agents.ChatEval import ChatEval
# from benchmark.src.agents.EvoAgent import EvoAgent

# List available agent systems for easy reference
AVAILABLE_AGENT_SYSTEMS = {
    "single_agent": "A single LLM agent solving problems directly",
    "supervisor_mas": "A supervisor-based multi-agent system where a supervisor coordinates specialized agents",
    "swarm": "A swarm-based multi-agent system where multiple agents work independently and results are aggregated",
    "agentverse": "A multi-agent system that uses a recruiter to create specialized agents for different aspects of a problem",
    "chateval": "A multi-agent system that uses debate to generate a better answer",
    "evoagent": "An evolutionary agent system that improves over generations using LLM-based operations",
    "mock_triple_agent": "A mock multi-agent system that uses a triple agent to solve problems with mock tools",
    "metagpt": "A multi-agent system that uses SOPs to generate codes",
    "jarvis": "A multi-agent system that uses linear task planning and execution process",
}

__all__ = [
    "AgentSystem",
    "AgentSystemRegistry",
    "create_agent_system",
    "SupervisorMAS",
    "SwarmSystem",
    "SingleAgent",
    "AgentVerse",
    "ChatEval",
    "EvoAgent",
    "MockTripleAgentSystem",
    "MetaGPT",
    "jarvis",
    "AVAILABLE_AGENT_SYSTEMS",
]
