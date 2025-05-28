import time
import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry
from typing import TypedDict, List, Dict, Any

class PRD(TypedDict):
    """Product Requirements Document"""
    title: str
    description: str
    features: List[str]
    requirements: List[str]

class ArchitectureDesign(TypedDict):
    """System Architecture Design"""
    system_architecture: str
    technology_stack: List[str]
    architecture_diagram: str
    interface_specifications: Dict[str, Any]

class TaskBreakdown(TypedDict):
    """Task Breakdown"""
    tasks: List[Dict[str, Any]]
    task_assignments: Dict[str, List[str]]
    task_priorities: Dict[str, int]
    task_dependencies: Dict[str, List[str]]

class CodeImplementation(TypedDict):
    """Code Implementation"""
    code: str
    implementation_details: str
    bug_fixes: List[str]
    performance_optimizations: List[str]

class TestResults(TypedDict):
    """Test Results"""
    test_cases: List[Dict[str, Any]]
    test_execution_results: Dict[str, bool]
    bugs_found: List[str]
    improvement_suggestions: List[str] 


@dataclass
class Agent:
    """Base Agent class containing key attributes such as name, profile, goals, constraints and description"""
    name: str
    description: str
    goals: List[str]
    constraints: List[str]
    role: str
    system_prompt: str
    memory: Dict[str, Any] = None
    llm: Any = field(init=False, repr=False)
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = {
                "messages": [],
                "knowledge": {},
                "tasks": [],
                "completed_tasks": []
            }
        self.llm = ChatOpenAI(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )
    
    def add_to_memory(self, key: str, value: Any):
        """Add information to agent's memory"""
        if key not in self.memory:
            self.memory[key] = []
        self.memory[key].append(value)
    
    def get_from_memory(self, key: str) -> Any:
        """Retrieve information from agent's memory"""
        return self.memory.get(key, [])
    
    def clear_memory(self, key: str = None):
        """Clear agent's memory"""
        if key:
            self.memory[key] = []
        else:
            self.memory = {
                "messages": [],
                "knowledge": {},
                "tasks": [],
                "completed_tasks": []
            }


class MetaGPT(AgentSystem):
    """Multi-agent system based on Standard Operating Procedures (SOP)"""
    
    def __init__(self, name: str = None, config: Dict[str, Any] = None):
        """
        Initialize SOPMAS system
        
        Args:
            name: System name
            config: Configuration parameters
        """
        super().__init__(name, config)
        
        # Initialize agents by calling _create_agents
        create_agents_result = self._create_agents()
        # self.agents is now set within _create_agents.
        # The 'workers' key in create_agents_result is for ToolIntegrationWrapper.

        # Initialize message queue
        self.message_queue = []
        
        # Initialize task status
        self.task_status = {
            "current_task": None,
            "task_history": [],
            "iteration_count": 0,
            "max_iterations": self.config.get("max_iterations", 3)
        }
        
        # Initialize LLM for MetaGPT (e.g., for summarization or tasks not tied to a specific role's LLM)
        # This LLM is distinct from the LLMs within each Agent.
        self.llm = ChatOpenAI(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        )
        
        # Initialize message history
        self.message_history = []
    
    def _create_agents(self) -> Dict[str, List[Agent]]:
        """Initialize all agents in the system and return them for TIW."""
        agents_dict = {}
        
        # Product Manager (PM)
        agents_dict["product_manager"] = Agent(
    name="Product Manager",
    description="Responsible for analysing the HumanEval prompt, defining requirements and acceptance criteria.",
    goals=[
        "Clarify user story", "Define acceptance criteria", "Produce task list"
    ],
    constraints=[
        "Keep requirements concise", "Avoid technical implementation details"
    ],
    role="product_manager",
    system_prompt=(
        "You are **Product Manager**.\n\n"
        "### Workflow\n"
        "1. Read the HumanEval problem statement.\n"
        "2. Summarise the *user story* in one sentence.\n"
        "3. List *acceptance criteria* that the final code must satisfy.\n"
        "4. Break work into *tasks* for Architect / Engineer / QA.\n\n"
        "### Output Markdown Template  \n"
        "```\n"
        "## User Story\n"
        "- …\n\n"
        "## Acceptance Criteria\n"
        "- …\n"
        "- …\n\n"
        "## Task List\n"
        "- A1 Architect: …\n"
        "- E1 Engineer: …\n"
        "- QA1 QA Engineer: …\n"
        "```\n"
        "Use exactly these headings; do **not** output JSON."
    )
)

# Architect
        agents_dict["architect"] = Agent(
    name="Architect",
    description="Responsible for high-level design and defining module interfaces.",
    goals=[
        "Provide architecture diagram", "Define module interfaces", "Choose data structures"
    ],
    constraints=[
        "Follow PM requirements", "Design for readability & efficiency"
    ],
    role="architect",
    system_prompt=(
        "You are **Architect**.\n\n"
        "### Workflow\n"
        "1. Receive requirements & tasks from Product Manager.\n"
        "2. Outline the solution architecture.\n"
        "3. Specify *function signatures* and key data structures.\n"
        "4. Recommend algorithms / complexity.\n\n"
        "### Output Markdown Template  \n"
        "```\n"
        "## High-Level Design\n"
        "- …\n\n"
        "## Module Interfaces\n"
        "```python\n"
        "# function stubs / interfaces here\n"
        "```\n\n"
        "## Design Rationale\n"
        "- Time / space complexity: …\n"
        "- Trade-offs: …\n"
        "```\n"
        "Everything that must be coded later goes in the python block under *Module Interfaces*."
    )
)

# Project Manager (PMgr)
        agents_dict["project_manager"] = Agent(
    name="Project Manager",
    description="Responsible for planning, scheduling and risk management.",
    goals=[
        "Create timeline", "Track progress", "Mitigate risks"
    ],
    constraints=[
        "Timeline must be short (<=10 steps)", "Highlight critical path"
    ],
    role="project_manager",
    system_prompt=(
        "You are **Project Manager**.\n\n"
        "### Workflow\n"
        "1. Collect task list from Product Manager.\n"
        "2. Produce a concise timeline (max 10 steps).\n"
        "3. Identify risks and mitigation strategies.\n"
        "4. Assign owners.\n\n"
        "### Output Markdown Template  \n"
        "```\n"
        "## Timeline\n"
        "| Step | Owner | Description |\n"
        "|------|-------|-------------|\n"
        "| 1 | Architect | … |\n"
        "| 2 | Engineer  | … |\n"
        "| … | … | … |\n\n"
        "## Risks & Mitigations\n"
        "- *Risk*: …  \n"
        "  *Mitigation*: …\n"
        "- *Risk*: …  \n"
        "  *Mitigation*: …\n"
        "```\n"
        "Stick to the table format for the timeline; it makes parsing easy."
    )
)
        # Engineer
        agents_dict["engineer"] = Agent(
    name="Engineer",
    description="Responsible for coding and implementation according to architect design and PM tasks.",
    goals=[
        "Write code", "Implement features", "Fix bugs", "Optimize performance"
    ],
    constraints=[
        "Follow architecture", "Follow PEP-8", "Write unit-test-ready code"
    ],
    role="engineer",
    system_prompt=(
        "You are **Engineer**.\n\n"
        "### Workflow\n"
        "1. Pick up the task description (HumanEval prompt).\n"
        "2. Provide a *concise* explanation of your approach.\n"
        "3. Write Python code **inside a fenced block** exactly once.\n"
        "4. List any bug-fixes or optimisations you applied.\n\n"
        "### Output Markdown Template  \n"
        "```\n"
        "## Explanation\n"
        "- …\n\n"
        "## Code\n"
        "```python\n"
        "# your solution here\n"
        "```\n\n"
        "## Bug Fixes & Optimisations\n"
        "- …\n"
        "```\n"
        "Use this template verbatim so the evaluator can locate the ```python code block```."
    )
)

# QA Engineer
        agents_dict["qa_engineer"] = Agent(
    name="QA Engineer",
    description="Responsible for testing and delivering the final validated code.",
    goals=[
        "Design tests", "Run tests", "Report bugs", "Return final code"
    ],
    constraints=[
        "Tests must cover edge cases", "Return final working code block"
    ],
    role="qa_engineer",
    system_prompt=(
        "You are **QA Engineer**.\n\n"
        "### Workflow\n"
        "1. Receive candidate code from Engineer.\n"
        "2. Write additional tests inline if needed.\n"
        "3. Fix any discovered issues (light edits only).\n"
        "4. **Return the final, tested code in a single ```python block** under `## Final Code`.\n\n"
        "### Output Markdown Template  \n"
        "```\n"
        "## Test Report\n"
        "- Passed cases: …\n"
        "- Failed cases: … / None\n\n"
        "## Final Code\n"
        "```python\n"
        "# final solution here (will be executed by evaluator)\n"
        "```\n"
        "```\n"
        "Evaluator will run ONLY the code in `## Final Code`."
    )
)


        self.agents = agents_dict
        return {"workers": list(agents_dict.values())}
    
    def _publish_message(self, from_agent: str, to_agent: str, message_type: str, content: Any):
        """Publish message to message queue"""
        message = {
            "from": from_agent,
            "to": to_agent,
            "type": message_type,
            "content": content,
            "timestamp": time.time()
        }
        self.message_queue.append(message)
        
        # Add message to receiver's memory
        if to_agent in self.agents:
            self.agents[to_agent].add_to_memory("messages", message)
    
    def _subscribe_messages(self, agent_name: str, message_type: str = None) -> List[Dict[str, Any]]:
        """Subscribe to messages in message queue"""
        messages = []
        for message in self.message_queue:
            if message["to"] == agent_name and (message_type is None or message["type"] == message_type):
                messages.append(message)
        return messages
    
    def _run_agent_task(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent task"""
        agent = self.agents[agent_name]
        
        # Build messages
        messages = [
            SystemMessage(content=agent.system_prompt),
            HumanMessage(content=f"Task information: {json.dumps(task, ensure_ascii=False)}")
        ]

        # Get corresponding schema
        schema_map = {
        "product_manager": None,
        "architect": None,
        "project_manager": None,
        "engineer": None,
        "qa_engineer": None
        }
        # Get corresponding schema
        schema = schema_map.get(agent_name)
        
        # Use the agent's own LLM instance
        agent_llm = agent.llm
        
        # Initialize LLM, use schema if agent needs structured output
        if schema:
            # Bind structured output to the agent's LLM for this call
            invokable_llm = agent_llm.with_structured_output(schema=schema, include_raw=True)
        else:
            invokable_llm = agent_llm
        
        # Run LLM
        response = invokable_llm.invoke(messages)
        
        if schema:
            # For structured output
            content = response["parsed"]
            raw_response = response["raw"]
            
            # Get usage_metadata
            usage_metadata = raw_response.usage_metadata if hasattr(raw_response, "usage_metadata") else None
            
            # Build result
            result = {
                "content": content,
                "agent": agent_name
            }
            
            # Create AIMessage and save to history
            ai_message = AIMessage(content=str(content))
            ai_message.name = agent_name
            if usage_metadata:
                ai_message.usage_metadata = usage_metadata
            self.message_history.append(ai_message)
            
        else:
            # For unstructured output
            result = {
                "content": response.content,
                "agent": agent_name
            }
            
            # Get usage_metadata
            usage_metadata = response.usage_metadata if hasattr(response, "usage_metadata") else None
            
            # Save to message history
            response.name = agent_name
            self.message_history.append(response)
        
        # Publish message
        message = {
            "content": result,
            "usage_metadata": usage_metadata  # Use obtained usage_metadata directly
        }
        self._publish_message(agent_name, "system", "task_result", message)
        
        return result

    def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task"""
        # Update current task
        self.task_status["current_task"] = task
        
        # Product Manager processes requirements
        product_manager_result = self._run_agent_task("product_manager", task)
        
        # Architect and Project Manager work
        architect_result = self._run_agent_task("architect", product_manager_result)
        project_manager_result = self._run_agent_task("project_manager", {
            "product_manager_result": product_manager_result
        })
        
        # Update Project Manager's result with Architect's result
        project_manager_result = self._run_agent_task("project_manager", {
            "product_manager_result": product_manager_result,
            "architect_result": architect_result
        })
        
        # Developer works
        developer_result = self._run_agent_task("engineer", {
            "product_manager_result": product_manager_result,
            "architect_result": architect_result,
            "project_manager_result": project_manager_result
        })
        
        # Tester works
        tester_result = self._run_agent_task("qa_engineer", {
            "product_manager_result": product_manager_result,
            "architect_result": architect_result,
            "project_manager_result": project_manager_result,
            "developer_result": developer_result
        })
        
        # Return results
        result = {
            "product_manager_result": product_manager_result,
            "architect_result": architect_result,
            "project_manager_result": project_manager_result,
            "developer_result": developer_result,
            "tester_result": tester_result
        }
        
        return result

    def run_agent(self, problem: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run MAS system
        
        Args:
            problem: Problem data
            
        Returns:
            Execution results
        """
        # Record start time
        start_time = time.time()
        
        try:
            # Clear message history
            self.message_history = []
            
            # Process task
            result = self._process_task(problem)
            
            # Record end time
            end_time = time.time()
            
            # Build final answer
            # Extract final answer from tester's result
            final_answer = ""
            if "tester_result" in result and "content" in result["tester_result"]:
                final_answer = result["tester_result"]["content"]
            else:
                # If no test result, use engineer's result
                if "developer_result" in result and "content" in result["developer_result"]:
                    final_answer = result["developer_result"]["content"]
                else:
                    # If still no result, use combination of all results
                    final_answer = json.dumps(result, ensure_ascii=False, indent=2)
            
            # Return result, ensure all necessary fields are included
            return {
                "result": result,
                "execution_time": end_time - start_time,
                "messages": [msg for msg in self.message_history if hasattr(msg, 'usage_metadata')],  # Only return messages with usage_metadata
                "extracted_answer": final_answer
            }
            
        except Exception as e:
            print(f"Error occurred during execution: {str(e)}")
            # Return result containing error information
            return {
                "result": {"error": str(e)},
                "execution_time": 0,
                "messages": self.message_history,  # Return collected messages even if error occurs
                "extracted_answer": f"Execution error: {str(e)}"
            }

    def _need_iteration(self, qa_result: Dict[str, Any]) -> bool:
        """Check if iteration is needed"""
        # If QA found bugs, iteration is needed
        if "bugs" in qa_result and qa_result["bugs"]:
            return True
        
        # If QA provided improvement suggestions, iteration is needed
        if "improvements" in qa_result and qa_result["improvements"]:
            return True
        
        return False


# Register MetaGPT system
AgentSystemRegistry.register(
    "metagpt",
    MetaGPT,
    max_iterations=3
)

if __name__ == "__main__":
    # Create MetaGPT instance
    config = {
        "max_iterations": 3
    }
    metagpt = MetaGPT(name="Test System", config=config)
    
    # Create test problem
    test_problem = {
        "id": "test_001",
        "type": "code_generation",
        "description": "Create a simple Python function to add two numbers",
        "requirements": [
            "Function name should be add_numbers",
            "Accept two parameters a and b",
            "Return the result of a+b",
            "Include appropriate type hints",
            "Include docstring"
        ]
    }
    
    # Run test
    try:
        print("Starting MetaGPT system test...")
        result = metagpt.run_agent(test_problem)
        
        print("\nTest Results:")
        print("-" * 50)
        print(f"Execution time: {result['execution_time']:.2f}s")
        
        print("\nTask Execution Results:")
        print("-" * 50)
        print(result['extracted_answer'])
        print(result['result']['tester_result']['content'])
        
        print("\nMessage History:")
        print("-" * 50)
        for msg in result.get("messages", []):
            print(f"\n{msg.name}'s message:")
            print("-" * 20)
            if hasattr(msg, "content"):
                print(msg.content)
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                print(f"\nToken usage:")
                print(json.dumps(msg.usage_metadata, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error occurred during test: {str(e)}") 
