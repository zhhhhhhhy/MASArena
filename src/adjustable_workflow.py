"""
AdjustableWorkflow: Extends the BaseWorkflow with runtime adjustment capabilities.

This implementation allows for dynamic modification of agent connections (edges)
by using a ConnectionManager to define the routing between agents.
"""

import os
import re
import logging
from typing import Dict, List, Literal, TypedDict, Union, Any, Optional, cast, Set
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

from src.base_workflow import BaseWorkflow, logger
from src.connection_manager import ConnectionManager

class AdjustableWorkflow(BaseWorkflow):
    """
    Extends the BaseWorkflow with runtime adjustment capabilities.
    
    This workflow uses a ConnectionManager to define the connections between agents,
    allowing for dynamic modification of agent routing at runtime.
    """
    
    def __init__(self, agent_names: List[str], model_name: Optional[str] = None, connection_manager: Optional[ConnectionManager] = None):
        """
        Initialize the AdjustableWorkflow.
        
        Args:
            agent_names: List of agent names to include in the network
            model_name: Optional model name to use (defaults to environment variable)
            connection_manager: Optional ConnectionManager defining agent connections
        """
        # Initialize the connection manager if not provided
        self.connection_manager = connection_manager or ConnectionManager(agent_names)
        
        # Initialize the LLM
        self._init_llm(model_name)
        
        # Store agent names
        self.agent_names = agent_names
        
        # Build the graph
        self._build_graph()
        
        # Log the connection graph
        self._log_connection_graph()
    
    def _init_llm(self, model_name: Optional[str] = None):
        """Initialize the language model."""
        # Load environment variables
        load_dotenv()
        
        # Get the model name from the environment variable if not provided
        model = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        base_url = os.getenv("BASE_URL")
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model=model,
            temperature=0.7,
            base_url=base_url if base_url else None 
        )
    
    def _build_graph(self):
        """Build the graph based on the current connection manager."""
        self.graph = super()._build_graph()
    
    def _log_connection_graph(self):
        """Log the current connection graph structure."""
        logger.info("Connection Graph Structure:", extra={"agent_name": "System"})
        
        for agent in self.agent_names:
            # Get all outgoing connections for this agent
            connections = self.connection_manager.get_allowed_next_agents(agent)
            logger.info(f"{agent} can route to: {', '.join(str(conn) for conn in connections)}", extra={"agent_name": "System"})
    
    def _create_agent_node(self, agent_name: str):
        """
        Create an agent node function for the graph that respects the connection graph.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            A function that processes the state and returns a command
        """
        # Define the possible destinations for type hinting
        possible_destinations = self.agent_names + [END]
        command_type = cast(Any, Command[Literal[tuple(possible_destinations)]])
        
        def agent_node(state: MessagesState) -> command_type:
            """Agent node function that processes state and decides where to go next."""
            # Extract the conversation history
            messages = state["messages"]
            
            # Initialize turn count if it doesn't exist
            if "turn_count" not in state:
                state["turn_count"] = 0
            
            # Increment turn count
            state["turn_count"] += 1
            
            # Count how many times this agent has been called
            agent_call_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and msg.content.startswith(f"[{agent_name}]"))
            
            # Log that this agent is being called
            logger.info(f"Calling {agent_name} (call #{agent_call_count + 1})", extra={"agent_name": agent_name})
            
            # Get the allowed next agents from the connection manager
            allowed_next_agents = self.connection_manager.get_allowed_next_agents(agent_name)
            
            # Create a system prompt for the agent
            system_prompt = f"""You are {agent_name}, part of a multi-agent system.
            
Your task is to process the conversation history and decide which agent should be called next,
or if the conversation should end.

Available agents: {', '.join([name for name in allowed_next_agents if name != END])}
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
                if next_name.lower() == "__end__" and END in allowed_next_agents:
                    next_agent = END
                else:
                    # Find the closest matching agent name among allowed next agents
                    for name in allowed_next_agents:
                        if name != END and name.lower() == next_name.lower():
                            next_agent = name
                            break
            
            # If no explicit NEXT format, fall back to mention detection
            if next_agent is None:
                for name in allowed_next_agents:
                    if name == END:
                        if "__end__" in response_text.lower():
                            next_agent = END
                            break
                    else:
                        # Look for the agent name as a whole word
                        if re.search(r'\b' + re.escape(name.lower()) + r'\b', response_text.lower()):
                            next_agent = name
                            break
            
            # Default to END if no agent was found, if this agent has been called too many times,
            # or if the agent selected is not in the allowed next agents
            if next_agent is None or agent_call_count >= 3:
                # If END is an allowed next agent, use it
                if END in allowed_next_agents:
                    next_agent = END
                else:
                    # Otherwise, pick the first allowed agent
                    for name in allowed_next_agents:
                        if name != agent_name:
                            next_agent = name
                            break
            
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
                update={"messages": state["messages"] + [ai_message], "turn_count": state["turn_count"]}
            )
        
        return agent_node
    
    def update_connection_manager(self, connection_manager: ConnectionManager):
        """
        Update the connection manager at runtime.
        
        Args:
            connection_manager: The new ConnectionManager
        """
        # Validate the connection manager
        self._validate_connection_manager(connection_manager)
        
        # Update the connection manager
        self.connection_manager = connection_manager
        
        # Rebuild the graph to reflect the new connections
        self._build_graph()
        
        # Log the updated graph
        logger.info("Connection graph updated", extra={"agent_name": "System"})
        self._log_connection_graph()
    
    def _validate_connection_manager(self, connection_manager: ConnectionManager):
        """
        Validate that the connection manager contains all agents and has valid connections.
        
        Args:
            connection_manager: The ConnectionManager to validate
            
        Raises:
            ValueError: If the ConnectionManager is invalid
        """
        # Check that all agents are in the graph
        for agent in self.agent_names:
            if agent not in connection_manager.agent_names:
                raise ValueError(f"Agent '{agent}' is missing from the connection manager")
    
    def add_connection(self, source: str, target: str):
        """
        Add a connection between two agents.
        
        Args:
            source: The source agent
            target: The target agent
        """
        self.connection_manager.add_connection(source, target)
        logger.info(f"Added connection: {source} → {target}", extra={"agent_name": "System"})
        
        # Rebuild the graph to reflect the new connection
        self._build_graph()
    
    def remove_connection(self, source: str, target: str):
        """
        Remove a connection between two agents.
        
        Args:
            source: The source agent
            target: The target agent
        """
        self.connection_manager.remove_connection(source, target)
        logger.info(f"Removed connection: {source} → {target}", extra={"agent_name": "System"})
        
        # Rebuild the graph to reflect the removed connection
        self._build_graph()
    
    def set_connections(self, agent_name: str, targets: List[str]):
        """
        Set the connections for an agent, replacing any existing connections.
        
        Args:
            agent_name: The agent name
            targets: List of target agent names
        """
        self.connection_manager.set_connections(agent_name, targets)
        logger.info(f"Set connections for {agent_name}: {', '.join(str(target) for target in targets)}", extra={"agent_name": "System"})
        
        # Rebuild the graph to reflect the new connections
        self._build_graph()
    
    def reset_to_fully_connected(self):
        """Reset the graph to a fully connected configuration."""
        self.connection_manager.reset_to_fully_connected()
        logger.info("Reset to fully connected graph", extra={"agent_name": "System"})
        
        # Rebuild the graph to reflect the fully connected configuration
        self._build_graph()
        
        self._log_connection_graph()
    
    def get_connections(self, agent_name: str) -> List[str]:
        """
        Get all outgoing connections for an agent.
        
        Args:
            agent_name: The agent to get connections for
            
        Returns:
            A list of agent names that this agent can route to
        """
        return self.connection_manager.get_allowed_next_agents(agent_name)
    
    def visualize_connections(self, title: str = "Agent Connection Graph", output_file: str = "connection_graph.png"):
        """
        Visualize the connection graph and save it to a file.
        
        Args:
            title: The title of the plot
            output_file: The file to save the visualization to
            
        Returns:
            The path to the saved visualization
        """
        return self.connection_manager.visualize(title, output_file)
    
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the workflow with the given state.
        
        Args:
            state: The initial state
            
        Returns:
            The final state
        """
        return self.graph.invoke(state) 