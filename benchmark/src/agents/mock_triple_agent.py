from benchmark.src.agents.base import AgentSystem
from typing import Dict, Any
from unittest.mock import MagicMock
from benchmark.src.agents.base import AgentSystemRegistry

class MockTripleAgentSystem(AgentSystem):
    """
    A mock agent system with three agents, each with mock math/search tools.
    Used for testing tool integration and tool calling logging.
    """
    def __init__(self, name="mock_triple_agent", config=None):
        super().__init__(name=name, config=config or {})
        self.create_agents_called = False
        self.workers = []

    def run_agent(self, problem, **kwargs):
        # Simulate multi-agent creation and tool assignment
        result = self._create_agents(problem)
        self.workers = result["workers"]
        responses = []
        for i, worker in enumerate(self.workers):
            # Each worker uses its LLM to generate a response
            response = worker.llm.invoke(problem["problem"]) if hasattr(worker, "llm") else f"Worker {i} response"
            responses.append(response)
        return {"messages": responses}

    def _create_agents(self, problem_input, feedback=None):
        self.create_agents_called = True
        # Create 3 mock worker agents with specific agent names matching tool_assignment
        agent_names = ["MathAgent", "SearchAgent", "ReasoningAgent"]
        self.workers = [MagicMock() for _ in range(3)]
        for i, worker in enumerate(self.workers):
            # Assign agent names that match the tool_assignment from mock_mcp_config.json
            if i < len(agent_names):
                worker.name = agent_names[i]
            else:
                worker.name = f"worker_{i}"
            # The tool integration system will assign tools and bind them to worker.llm
            # Here, we mock an llm with a .invoke() method for demonstration
            worker.llm = MagicMock()
            worker.llm.invoke.return_value = f"LLM response for {worker.name} (tools: {getattr(worker, 'tools', None)})"
        return {"workers": self.workers}

    def evaluate(self, problem, **kwargs):
        result = self.run_agent(problem, **kwargs)
        return {"score": 1, "answer": "mock answer", "messages": result["messages"]}

# Register the new MockTripleAgentSystem in the AgentSystemRegistry for use in the benchmark CLI
AgentSystemRegistry.register("mock_triple_agent", MockTripleAgentSystem)