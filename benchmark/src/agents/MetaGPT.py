import time
import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
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
    
    def __post_init__(self):
        if self.memory is None:
            self.memory = {
                "messages": [],
                "knowledge": {},
                "tasks": [],
                "completed_tasks": []
            }
    
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
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Initialize message queue
        self.message_queue = []
        
        # Initialize task status
        self.task_status = {
            "current_task": None,
            "task_history": [],
            "iteration_count": 0,
            "max_iterations": self.config.get("max_iterations", 3)
        }
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7
        )
        
        # Initialize message history
        self.message_history = []
    
    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents in the system"""
        agents = {}
        
        # Product Manager
        agents["product_manager"] = Agent(
            name="Product Manager",
            description="Responsible for requirement analysis, market research, writing PRD, and defining product goals and user stories.",
            goals=[
                "Analyze user requirements",
                "Conduct market research",
                "Write Product Requirements Document (PRD)",
                "Define product goals and user stories"
            ],
            constraints=[
                "Must consider actual user needs",
                "Must consider market feasibility",
                "Must clarify core product value"
            ],
            role="product_manager",
            system_prompt="""You are a professional Product Manager responsible for requirement analysis, market research, writing PRD, and defining product goals and user stories.
Your workflow is:
1. Receive user requirements
2. Conduct requirement analysis
3. Perform market research
4. Write Product Requirements Document (PRD)
5. Define product goals and user stories

Please ensure your output includes:
1. Requirement analysis results
2. Market research results
3. Product Requirements Document (PRD)
4. Product goals and user stories

Your output should be well-structured for other roles to understand and execute."""
        )
        
        # Architect
        agents["architect"] = Agent(
            name="Architect",
            description="Responsible for system design, including technology selection, system architecture diagram creation, and interface definition.",
            goals=[
                "Design system architecture",
                "Select technology stack",
                "Create system architecture diagram",
                "Define interface specifications"
            ],
            constraints=[
                "Must consider system scalability",
                "Must consider system maintainability",
                "Must consider system performance"
            ],
            role="architect",
            system_prompt="""You are a professional Architect responsible for system design, including technology selection, system architecture diagram creation, and interface definition.
Your workflow is:
1. Receive Product Requirements Document (PRD)
2. Design system architecture
3. Select technology stack
4. Create system architecture diagram
5. Define interface specifications

Please ensure your output includes:
1. System architecture design
2. Technology stack selection results
3. System architecture diagram
4. Interface specifications

Your output should be well-structured for other roles to understand and execute."""
        )
        
        # Project Manager
        agents["project_manager"] = Agent(
            name="Project Manager",
            description="Responsible for project management and task breakdown, breaking complex tasks into smaller subtasks and assigning them to engineers.",
            goals=[
                "Break down tasks",
                "Assign tasks",
                "Track task progress",
                "Coordinate team members"
            ],
            constraints=[
                "Must ensure reasonable task allocation",
                "Must ensure task feasibility",
                "Must ensure task priority"
            ],
            role="project_manager",
            system_prompt="""You are a professional Project Manager responsible for project management and task breakdown, breaking complex tasks into smaller subtasks and assigning them to engineers.
Your workflow is:
1. Receive PRD and system architecture design
2. Break down tasks
3. Assign tasks
4. Track task progress
5. Coordinate team members

Please ensure your output includes:
1. Task breakdown results
2. Task assignment results
3. Task progress tracking
4. Team coordination results

Your output should be well-structured for other roles to understand and execute."""
        )
        
        # Engineer
        agents["engineer"] = Agent(
            name="Engineer",
            description="Responsible for code writing and implementation, developing according to architect's design and project manager's task assignments.",
            goals=[
                "Write code",
                "Implement features",
                "Fix bugs",
                "Optimize performance"
            ],
            constraints=[
                "Must follow architecture design",
                "Must follow coding standards",
                "Must ensure code quality"
            ],
            role="engineer",
            system_prompt="""You are a professional Engineer responsible for code writing and implementation, developing according to architect's design and project manager's task assignments.
Your workflow is:
1. Receive task assignments
2. Write code
3. Implement features
4. Fix bugs
5. Optimize performance

Please ensure your output includes:
1. Code implementation
2. Feature description
3. Fixed bugs
4. Performance optimization results

Your output should be well-structured for other roles to understand and execute."""
        )
        
        # QA Engineer
        agents["qa_engineer"] = Agent(
            name="QA Engineer",
            description="Responsible for testing and quality assurance, ensuring code correctness and stability.",
            goals=[
                "Design test cases",
                "Execute tests",
                "Find bugs",
                "Provide improvement suggestions"
            ],
            constraints=[
                "Must ensure comprehensive testing",
                "Must ensure test accuracy",
                "Must ensure test reproducibility"
            ],
            role="qa_engineer",
            system_prompt="""You are a professional QA Engineer responsible for testing and quality assurance, ensuring code correctness and stability.
Your workflow is:
1. Receive code implementation
2. Design test cases
3. Execute tests
4. Find bugs
5. Provide improvement suggestions

Please ensure your output includes:
1. Test case design
2. Test execution results
3. Found bugs
4. Improvement suggestions

Your output should be well-structured for other roles to understand and execute."""
        )
        
        return agents
    
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
        
        # Run LLM directly
        # Choose schema based on agent type
        schema_map = {
            "product_manager": PRD,
            "architect": ArchitectureDesign,
            "project_manager": TaskBreakdown,
            "engineer": CodeImplementation,
            "qa_engineer": TestResults
        }
        
        # Get corresponding schema
        schema = schema_map.get(agent_name)
        
        # Initialize LLM, use schema if agent needs structured output
        if schema:
            llm = ChatOpenAI(
                model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                base_url=os.getenv("BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7
            ).with_structured_output(schema=schema, include_raw=True)
        else:
            llm = ChatOpenAI(
                model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                base_url=os.getenv("BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.7
            )
        
        # Run LLM
        response = llm.invoke(messages)
        
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
