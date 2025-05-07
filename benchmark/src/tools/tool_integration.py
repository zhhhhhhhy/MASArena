from typing import Dict, Any, List, Optional
from benchmark.src.tools.tool_selector import ToolSelector
from benchmark.src.agents.base import AgentSystem

class ToolIntegrationWrapper(AgentSystem):
    """
    Wraps any AgentSystem to inject MCP-tool integration.
    Delegates all calls to `inner`, but intercepts:
      - Multi-agent systems: after they generate sub-agents, assign tools.
      - Single-agent systems: before run_agent, select top-k tools.
    """
    def __init__(self, inner: AgentSystem, mcp_servers: Dict[str, Any], mock: bool = False):
        """
        Initialize by wrapping an existing agent system.
        
        Args:
            inner: The agent system being wrapped
            mcp_servers: Dict mapping service names to server configs
            mock: Whether to run in mock mode (no actual MCP server calls)
        """
        # We delegate to inner instead of calling super().__init__
        self.inner = inner
        # Copy name and config from inner
        self.name = inner.name
        self.config = inner.config.copy()
        # Initialize tool manager on wrapped agent
        inner.config["use_mcp_tools"] = True
        inner.config["mcp_servers"] = mcp_servers
        inner.config["mock_mcp"] = mock
        
        # Initialize tool manager if it doesn't exist yet
        if not hasattr(inner, "tool_manager") or inner.tool_manager is None:
            inner.init_tool_manager(mcp_servers)
            
        # Build the selector once
        self.selector = ToolSelector(inner.tool_manager.get_tools())
        
        # Apply patches based on the type of agent system
        self._apply_patches()

    def select_tools_for_problem(self, problem: Any, num_agents: Optional[int] = None) -> Any:
        """
        Select or partition tools for a given problem. This method can be overridden for custom selection algorithms.
        For multi-agent, num_agents should be provided.
        """
        if num_agents is not None and num_agents > 1:
            # Multi-agent: partition tools
            if isinstance(problem, dict):
                problem_desc = problem.get("problem", "")
            else:
                problem_desc = str(problem)
            return self.selector.select_tools(
                problem_desc,
                num_agents=num_agents,
                overlap=False,
            )
        else:
            # Single-agent: select top tools
            if isinstance(problem, dict):
                problem_desc = problem.get("problem", "")
            else:
                problem_desc = str(problem)
            return self.selector.select_tools(problem_desc)
    
    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Delegate to inner agent's run_agent method and log tool calls if present."""
        result = self.inner.run_agent(problem, **kwargs)
        # Check for tool call in the result (LangChain AIMessage convention)
        tool_calls = None
        if isinstance(result, dict):
            # If result contains 'messages', check for tool_calls in each message
            messages = result.get("messages", [])
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"[ToolIntegration] Tool call detected in message: {msg.tool_calls}")
                if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                    print(f"[ToolIntegration] Additional tool message: {msg.additional_kwargs}")
            # Also check top-level result for tool_calls
            if 'tool_calls' in result:
                print(f"[ToolIntegration] Tool call detected in result: {result['tool_calls']}")
        return result
    
    def _apply_patches(self):
        """Apply the appropriate method patches based on agent system type."""
        # For MAS with _create_agents override
        if hasattr(self.inner, "_create_agents"):
            self._patch_multi_agent_system()
        else:
            # Single-agent fallback
            self._patch_single_agent_system()
    
    def _patch_multi_agent_system(self):
        """Patch a multi-agent system to distribute tools to workers."""
        # Bind to the original class-defined _create_agents to bypass any base patches
        orig = self.inner.__class__._create_agents.__get__(self.inner, self.inner.__class__)
        wrapper_self = self
        
        def patched_create(wrapped_self, problem_input, feedback=None):
            # Call the original _create_agents with both arguments
            result = orig(problem_input, feedback)
            workers = result.get("workers", [])
            
            if workers:
                # Check for explicit tool assignment rules
                assignment_rules = {}
                try:
                    assignment_rules = self.inner.tool_manager.get_tool_assignment_rules() or {}
                except Exception:
                    assignment_rules = {}
                # Determine tool partitions based on assignment rules if available
                if assignment_rules:
                    # Map tool names to tool dicts
                    all_tools = {tool["name"]: tool for tool in wrapper_self.selector.tools}
                    parts = []
                    for worker in workers:
                        worker_name = getattr(worker, "name", "unknown")
                        assigned_names = assignment_rules.get(worker_name, [])
                        worker_tools = []
                        for name in assigned_names:
                            if name in all_tools:
                                worker_tools.append(all_tools[name])
                            else:
                                print(f"[ToolIntegration] Warning: assigned tool '{name}' for worker '{worker_name}' not found")
                        parts.append(worker_tools)
                else:
                    # Fallback to unified selection method
                    parts = wrapper_self.select_tools_for_problem(problem_input, num_agents=len(workers))
                    
                # Assign tools to each worker
                for i, worker in enumerate(workers):
                    if i < len(parts):
                        worker_tools = parts[i]
                        tool_objs = [t.get("tool_object") for t in worker_tools if t.get("tool_object")]
                        worker_name = getattr(worker, "name", f"worker_{i}")
                        print(f"[ToolIntegration] Worker '{worker_name}' received {len(tool_objs)} tools: {', '.join([t.get('name') for t in worker_tools])}")
                        setattr(worker, "tools", worker_tools)
                        print(f"[ToolIntegration] Worker '{worker_name}' has attributes llm: {hasattr(worker, 'llm')}")
                        if hasattr(worker, "llm") and hasattr(worker.llm, "bind_tools"):
                            print(f"[ToolIntegration] Binding tools <{tool_objs}> to worker '{worker_name}'")
                            worker.llm = worker.llm.bind_tools(tool_objs)
            
            return result
        
        from types import MethodType
        self.inner._create_agents = MethodType(patched_create, self.inner)
        
        print(f"[ToolIntegration] Successfully patched {self.inner.name} for multi-agent tool distribution")
    
    def _patch_single_agent_system(self):
        """Patch a single-agent system to select tools before running."""
        orig_run = self.inner.run_agent
        wrapper_self = self
        
        def patched_run(wrapped_self, problem, **kwargs):
            # Use the unified selection method
            tools = wrapper_self.select_tools_for_problem(problem)
            tool_objs = [t["tool_object"] for t in tools if "tool_object" in t]
            # Assign tools to agent for logging/metadata
            setattr(wrapper_self.inner, "tools", tools)
            # If the agent has an LLM, bind the tools
            if hasattr(wrapper_self.inner, "llm") and hasattr(wrapper_self.inner.llm, "bind_tools"):
                wrapper_self.inner.llm = wrapper_self.inner.llm.bind_tools(tool_objs)
            return orig_run(problem, **kwargs)
        
        from types import MethodType
        self.inner.run_agent = MethodType(patched_run, self.inner)
        
        print(f"[ToolIntegration] Successfully patched {self.inner.name} for single-agent tool selection")

    def set_metrics_registry(self, registry):
        """Set metrics registry on inner agent system."""
        self.inner.set_metrics_registry(registry)
        return self

    def evaluate(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Delegate evaluation to inner agent system."""
        return self.inner.evaluate(problem, **kwargs)
    
    def __getattr__(self, name):
        """Delegate all other attribute access to inner agent system."""
        return getattr(self.inner, name) 