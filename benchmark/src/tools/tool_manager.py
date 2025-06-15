from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack


class ToolManager:
    """Manages MCP tool servers and provides tools to agent systems."""
    def __init__(self, mcp_servers: Dict[str, Dict] = None, mock_mode: bool = False, tool_assignment_rules: Optional[Dict[str, List[str]]] = None):
        self.mcp_servers = mcp_servers or {}
        self.client = None
        self.tools: List[Any] = []
        self.mock_mode = mock_mode
        # Optional mapping of agent names to lists of tool names (assignment rules)
        self.tool_assignment_rules: Dict[str, List[str]] = tool_assignment_rules or {}
        self._exit_stack = AsyncExitStack()
        
        # Immediately set mock tools if in mock mode
        if self.mock_mode:
            self.tools = self._create_mock_tools()

    async def __aenter__(self):
        # If in mock mode, just return self without connecting to any servers
        if self.mock_mode:
            return self

        # If not in mock mode, proceed with normal MCP client setup
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            if not self.mcp_servers:
                return self

            self.client = MultiServerMCPClient(self.mcp_servers)
            await self._exit_stack.enter_async_context(self.client)
            await self.load_tools()
            return self
        except ImportError as e:
            print(f"Error: langchain_mcp_adapters not found - {e}")
            # Fall back to mock mode if library not available
            self.mock_mode = True
            self.tools = self._create_mock_tools()
            return self
        except Exception as e:
            print(f"Error connecting to MCP servers: {e}")
            # Fall back to mock mode if connection fails
            self.mock_mode = True
            self.tools = self._create_mock_tools()
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    def _create_mock_tools(self) -> List[Any]:
        """Create mock tools for testing using LangChain's tools if available."""
        try:
            # Try to use LangChain's tool creation
            from langchain.tools import Tool, StructuredTool
            
            # Simple math functions
            def mock_add(a: float, b: float) -> float:
                """Add two numbers."""
                return a + b
                
            def mock_subtract(a: float, b: float) -> float:
                """Subtract b from a."""
                return a - b
                
            def mock_multiply(a: float, b: float) -> float:
                """Multiply two numbers."""
                return a * b
                
            def mock_math_solve(problem: str) -> str:
                """Solve a math problem."""
                return "42"
                
            def mock_search(query: str) -> str:
                """Search for information."""
                return f"Mock search results for: {query}"
                
            def mock_reason(question: str) -> str:
                """Reason about a question."""
                return f"After careful consideration of '{question}', the answer is 42."
            
            # Create structured tools with proper schema
            return [
                StructuredTool.from_function(
                    func=mock_add,
                    name="mock_add",
                    description="Mock add two numbers"
                ),
                StructuredTool.from_function(
                    func=mock_subtract,
                    name="mock_subtract",
                    description="Mock subtract two numbers"
                ),
                StructuredTool.from_function(
                    func=mock_multiply,
                    name="mock_multiply",
                    description="Mock multiply two numbers"
                ),
                StructuredTool.from_function(
                    func=mock_math_solve,
                    name="mock_math_solve",
                    description="Mock solve a math problem"
                ),
                StructuredTool.from_function(
                    func=mock_search,
                    name="mock_search",
                    description="Mock search for information"
                ),
                StructuredTool.from_function(
                    func=mock_reason,
                    name="mock_reason",
                    description="Mock reason about a question"
                ),
            ]
        except ImportError:
            # Fall back to simple mock objects if LangChain isn't available
            print("Warning: LangChain tools not available, using simple mock objects")
            
            class SimpleMockTool:
                def __init__(self, name, description, func):
                    self.name = name
                    self.description = description
                    self.func = func
                    
                def __call__(self, *args, **kwargs):
                    return self.func(*args, **kwargs)
            
            return [
                SimpleMockTool("mock_add", "Mock add two numbers", lambda a, b: a + b),
                SimpleMockTool("mock_subtract", "Mock subtract two numbers", lambda a, b: a - b),
                SimpleMockTool("mock_multiply", "Mock multiply two numbers", lambda a, b: a * b),
                SimpleMockTool("mock_math_solve", "Mock solve a math problem", lambda problem: "42"),
                SimpleMockTool("mock_search", "Mock search for information", 
                              lambda query: f"Mock search results for: {query}"),
                SimpleMockTool("mock_reason", "Mock reason about a question", 
                              lambda question: f"After careful consideration of '{question}', the answer is 42."),
            ]

    async def load_tools(self) -> List[Any]:
        """Load tools from MCP client or use mock tools."""
        if self.mock_mode:
            self.tools = self._create_mock_tools()
            return self.tools

        if not self.client:
            return []

        try:
            self.tools = self.client.get_tools()
            return self.tools
        except Exception as e:
            print(f"Error loading tools from MCP client: {e}")
            # Fall back to mock tools on error
            self.mock_mode = True
            self.tools = self._create_mock_tools()
            return self.tools

    def get_tools(self) -> List[Any]:
        """Get the list of loaded tools."""
        # Always return mock tools if in mock mode, even if client.get_tools() failed
        if self.mock_mode:
            return self._convert_tool_format(self._create_mock_tools())
        return self._convert_tool_format(self.tools)
    
    def _convert_tool_format(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Convert tools to the dictionary format expected by ToolSelector.
        
        This method handles different tool types:
        - LangChain StructuredTool objects
        - SimpleMockTool objects
        - Tool dictionaries
        
        And converts them to a common format:
        {
            "name": str,
            "description": str,
            "category": str,
            "tool_object": original_tool  # Reference to original tool object
        }
        """
        result = []
        
        for tool in tools:
            if isinstance(tool, dict):
                # Already in dictionary format, just ensure it has a category
                if "category" not in tool:
                    tool["category"] = self._infer_category(tool.get("name", ""), 
                                                           tool.get("description", ""))
                result.append(tool)
                continue
                
            # Handle LangChain StructuredTool objects
            if hasattr(tool, "name") and hasattr(tool, "description"):
                name = tool.name
                description = tool.description
                # Try to get schema info if available
                args_schema = getattr(tool, "args_schema", None)
                
                tool_dict = {
                    "name": name,
                    "description": description,
                    "category": self._infer_category(name, description),
                    "tool_object": tool  # Keep reference to original tool
                }
                
                if args_schema:
                    tool_dict["args_schema"] = str(args_schema)
                    
                result.append(tool_dict)
                continue
                
            # Fallback for unknown tool types
            result.append({
                "name": str(tool),
                "description": "Unknown tool type",
                "category": "general",
                "tool_object": tool
            })
            
        return result
    
    def _infer_category(self, name: str, description: str) -> str:
        """Infer a tool's category from its name and description."""
        name = name.lower()
        description = description.lower()
        
        # Match common patterns to categories
        if any(kw in name or kw in description for kw in ["math", "calc", "solve", "add", "subtract", "multiply", "divide"]):
            return "math"
        elif any(kw in name or kw in description for kw in ["search", "find", "lookup", "query"]):
            return "search"
        elif any(kw in name or kw in description for kw in ["reason", "logic", "deduce", "think"]):
            return "reasoning"
        elif any(kw in name or kw in description for kw in ["code", "program", "execute", "run"]):
            return "code"
        elif any(kw in name or kw in description for kw in ["data", "analyze", "process", "transform"]):
            return "data"
        else:
            return "general"

    def categorize_tools(self) -> Dict[str, List[Any]]:
        """Categorize tools by common prefixes or keywords."""
        categories = {
            "math": [],
            "search": [],
            "reasoning": [],
            "data": [],
            "general": []
        }
        
        for tool in self.get_tools():
            name = getattr(tool, 'name', '').lower()
            if 'math' in name or 'calc' in name or 'solve' in name:
                categories["math"].append(tool)
            elif 'search' in name or 'find' in name or 'lookup' in name:
                categories["search"].append(tool)
            elif 'reason' in name or 'logic' in name or 'deduce' in name:
                categories["reasoning"].append(tool)
            elif 'data' in name or 'analyze' in name or 'process' in name:
                categories["data"].append(tool)
            else:
                categories["general"].append(tool)
                
        return categories
        
    def get_tool_assignment_rules(self) -> Dict[str, List[str]]:
        """Return the tool assignment rules loaded from configuration."""
        return self.tool_assignment_rules

    @classmethod
    def from_config_file(cls, config_file_path: str, mock_mode: bool = False) -> "ToolManager":
        """Create a ToolManager instance from a configuration file."""
        import json
        try:
            with open(config_file_path, 'r') as f:
                config = json.load(f)
            # Separate tool_assignment rules from server configs
            tool_assignment = config.get("tool_assignment", {})
            # All other top-level keys are treated as MCP server configs
            mcp_servers = {k: v for k, v in config.items() if k != "tool_assignment"}
            return cls(mcp_servers, mock_mode=mock_mode, tool_assignment_rules=tool_assignment)
        except Exception as e:
            print(f"Error loading config from {config_file_path}: {e}")
            # Fallback to mock-only manager with no assignment rules
            return cls({}, mock_mode=True, tool_assignment_rules={}) 