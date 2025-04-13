"""
Multi-Agent Systems for Benchmarking

This package provides various agent system implementations for benchmarking.
"""

from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry, create_agent_system
from benchmark.src.agents.supervisor_mas import SupervisorMAS
from benchmark.src.agents.swarm import SwarmSystem
from benchmark.src.agents.single_agent import SingleAgent
from benchmark.src.agents.AgentVerse import AgentVerse

# List available agent systems for easy reference
AVAILABLE_AGENT_SYSTEMS = {
    "single_agent": "A single LLM agent solving problems directly",
    "supervisor_mas": "A supervisor-based multi-agent system where a supervisor coordinates specialized agents",
    "swarm": "A swarm-based multi-agent system where multiple agents work independently and results are aggregated",
    "agentverse": "A multi-agent system that uses a recruiter to create specialized agents for different aspects of a problem",
}

__all__ = [
    "AgentSystem",
    "AgentSystemRegistry",
    "create_agent_system",
    "SupervisorMAS",
    "SwarmSystem",
    "SingleAgent",
    "AgentVerse",
    "AVAILABLE_AGENT_SYSTEMS",
]
