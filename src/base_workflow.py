"""
BaseWorkflow: A base workflow using LangGraph to create a fully connected network for multiple agents.

This implementation creates a network where each agent can communicate with every other agent
and can decide which agent to call next.
"""

import os
import re
import logging
from typing import Dict, List, Literal, TypedDict, Union, Any, Optional, cast
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

# Load environment variables
load_dotenv()

# Configure colored logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on agent name."""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        'BOLD_RED': '\033[1;31m',
        'BOLD_GREEN': '\033[1;32m',
        'BOLD_YELLOW': '\033[1;33m',
        'BOLD_BLUE': '\033[1;34m',
        'BOLD_MAGENTA': '\033[1;35m',
        'BOLD_CYAN': '\033[1;36m',
        'BOLD_WHITE': '\033[1;37m',
    }
    
    # Map agent names to colors
    AGENT_COLORS = {
        'Researcher': COLORS['BOLD_BLUE'],
        'Critic': COLORS['BOLD_RED'],
        'Synthesizer': COLORS['BOLD_GREEN'],
        'Planner': COLORS['BOLD_YELLOW'],
        'Executor': COLORS['BOLD_MAGENTA'],
        'System': COLORS['BOLD_CYAN'],
    }
    
    def format(self, record):
        # Extract agent name from the record if it exists
        agent_name = getattr(record, 'agent_name', None)
        
        # Get the color for this agent
        if agent_name and agent_name in self.AGENT_COLORS:
            color = self.AGENT_COLORS[agent_name]
        else:
            color = self.COLORS['WHITE']
        
        # Format the message with color
        formatted_msg = super().format(record)
        return f"{color}{formatted_msg}{self.COLORS['RESET']}"

# Set up logger
logger = logging.getLogger("multi_agent")
logger.setLevel(logging.INFO)

# Create console handler with the colored formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(message)s'))
logger.addHandler(console_handler)

class BaseWorkflow:
    """
    A base workflow using LangGraph to create a fully connected network for multiple agents.
    Each agent can communicate with every other agent and decide which agent to call next.
    """
    
    def __init__(self, agent_names: List[str], model_name: Optional[str] = None):
        """
        Initialize the BaseWorkflow.
        
        Args:
            agent_names: List of agent names to include in the network
            model_name: Optional model name to use (defaults to environment variable)
        """
        self.agent_names = agent_names
        
        # Initialize the LLM
        model_name = model_name or os.getenv("MODEL_NAME", "gpt-4o-mini")
        base_url = os.getenv("BASE_URL")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            base_url=base_url if base_url else None
        )
        
        # Build the graph
        self.graph : StateGraph = self._build_graph()

        # save the graph as image
        try:
            self.graph.get_graph().draw_mermaid_png(output_file_path="workflow.png")
            logger.info("Workflow graph saved as workflow.png", extra={"agent_name": "System"})
        except Exception as e:
            logger.warning(f"Could not save workflow graph: {e}", extra={"agent_name": "System"})
    
    def _create_agent_node(self, agent_name: str):
        """
        Create an agent node function for the graph.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            A function that processes the state and returns a command
        """
        # Define the possible destinations for type hinting
        # This is needed for Command to work properly with type checking
        possible_destinations = self.agent_names + [END]
        command_type = cast(Any, Command[Literal[tuple(possible_destinations)]])
        
        def agent_node(state: MessagesState) -> command_type:
            """Agent node function that processes state and decides where to go next."""
            # Extract the conversation history
            messages = state["messages"]
            
            # Count how many times this agent has been called
            agent_call_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and msg.content.startswith(f"[{agent_name}]"))
            
            # Log that this agent is being called
            logger.info(f"Calling {agent_name} (call #{agent_call_count + 1})", extra={"agent_name": agent_name})
            
            # Create a system prompt for the agent
            system_prompt = f"""You are {agent_name}, part of a multi-agent system.
            
Your task is to process the conversation history and decide which agent should be called next,
or if the conversation should end.

Available agents: {', '.join([name for name in self.agent_names if name != agent_name])}
To end the conversation, respond with "__end__".

IMPORTANT: Be decisive about which agent to call next or whether to end the conversation.
Clearly state your decision at the end of your response in this format:
"NEXT: [agent_name]" or "NEXT: __end__"

This is call #{agent_call_count + 1} to you. If you've been called more than 3 times, consider ending the conversation.
"""
            
            # Prepare messages for the LLM
            llm_messages = [
                SystemMessage(content=system_prompt)
            ]
            
            # Add conversation history
            for msg in messages:
                llm_messages.append(msg)
            
            # Call the LLM
            response = self.llm.invoke(llm_messages)
            response_text = response.content
            
            # Parse the response to determine the next agent
            next_agent = None
            
            # First check for the explicit NEXT format
            next_pattern = re.compile(r"NEXT:\s*(\w+|__end__)", re.IGNORECASE)
            match = next_pattern.search(response_text)
            
            if match:
                next_name = match.group(1).strip()
                if next_name.lower() == "__end__":
                    next_agent = END
                else:
                    # Find the closest matching agent name
                    for name in self.agent_names:
                        if name.lower() == next_name.lower() and name != agent_name:
                            next_agent = name
                            break
            
            # If no explicit NEXT format, fall back to mention detection
            if next_agent is None:
                for name in self.agent_names + [END]:
                    if name == END:
                        if "__end__" in response_text.lower():
                            next_agent = END
                            break
                    else:
                        # Look for the agent name as a whole word
                        if name != agent_name and re.search(r'\b' + re.escape(name.lower()) + r'\b', response_text.lower()):
                            next_agent = name
                            break
            
            # Default to END if no agent was found or if this agent has been called too many times
            if next_agent is None or agent_call_count >= 3:
                next_agent = END
            
            # Create an AI message with the agent's response
            ai_message = AIMessage(content=f"[{agent_name}]: {response_text}")
            
            # Log the agent's response and next agent
            if next_agent == END:
                logger.info(f"{agent_name} response: {response_text}\n→ Ending conversation", extra={"agent_name": agent_name})
            else:
                logger.info(f"{agent_name} response: {response_text}\n→ Next: {next_agent}", extra={"agent_name": agent_name})
            
            # Return the command with the next agent to call
            return Command(
                goto=next_agent,
                update={"messages": state["messages"] + [ai_message]}
            )
        
        return agent_node
    
    def _build_graph(self) -> StateGraph:
        """
        Build the graph with all agents fully connected using Command objects.
        
        Returns:
            The compiled StateGraph
        """
        # Create the graph builder
        builder = StateGraph(MessagesState)
        
        # Add all agent nodes
        for agent_name in self.agent_names:
            builder.add_node(agent_name, self._create_agent_node(agent_name))
        
        # Connect START to the first agent
        builder.add_edge(START, self.agent_names[0])
        
        # We don't need to add edges between agents anymore since we're using Command objects
        # Each agent will decide where to go next using the Command return value
        
        # Log the graph structure
        logger.info("Graph structure:", extra={"agent_name": "System"})
        logger.info(f"Agents: {', '.join(self.agent_names)}", extra={"agent_name": "System"})
        logger.info(f"Starting with: {self.agent_names[0]}", extra={"agent_name": "System"})
        logger.info("Using Command objects for dynamic routing between agents", extra={"agent_name": "System"})
        
        # Compile the graph
        return builder.compile()
    
    def run(self, query: str):
        """
        Run the workflow with an initial query.
        
        Args:
            query: The initial query to start the conversation
            
        Returns:
            The final state after the workflow completes
        """
        # Log the query
        logger.info(f"Starting workflow with query: {query}", extra={"agent_name": "System"})
        
        # Create the initial state with the query
        initial_state = {"messages": [HumanMessage(content=query)]}
        
        try:
            # Run the graph with a try-except to catch recursion errors
            result = self.graph.invoke(initial_state, config={"recursion_limit": 10})
            logger.info("Workflow completed successfully", extra={"agent_name": "System"})
            return result
        except Exception as e:
            if "Recursion limit" in str(e):
                logger.warning(f"Warning: Reached recursion limit. Stopping execution.", extra={"agent_name": "System"})
                # Return the last state we have
                return initial_state
            else:
                # Re-raise other exceptions
                logger.error(f"Error running workflow: {e}", extra={"agent_name": "System"})
                raise e


# Example usage for debugging
if __name__ == "__main__":
    # Create a workflow with three agents
    workflow = BaseWorkflow(
        agent_names=["Researcher", "Critic", "Synthesizer"]
    )
    
    # Run the workflow with an initial query
    print("\n=== Starting Multi-Agent Workflow ===\n")
    query = "What are the key considerations for implementing a multi-agent system?"
    print(f"Query: {query}")
    
    try:
        result = workflow.run(query)
        
        # Print the conversation history
        print("\n=== Conversation History ===\n")
        for i, message in enumerate(result["messages"]):
            if i == 0:  # Skip the initial query as we already printed it
                continue
            print(f"{message.content}\n")
    except Exception as e:
        print(f"Error running workflow: {e}") 