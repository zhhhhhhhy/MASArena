"""
AdjustableWorkflow: Extends the BaseWorkflow with runtime adjustment capabilities.

This implementation allows for dynamic modification of agent connections (edges)
by using an external graph to define the routing between agents.
"""

import os
import re
import logging
import networkx as nx
from typing import Dict, List, Literal, TypedDict, Union, Any, Optional, cast, Set
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

from src.base_workflow import BaseWorkflow, logger

class ConfigWorkflow(BaseWorkflow):
    """
    Extends the BaseWorkflow with runtime adjustment capabilities.
    
    This workflow uses an external graph to define the connections between agents,
    allowing for dynamic modification of agent connections at runtime.
    """
    
    def __init__(self, agent_names: List[str], model_name: Optional[str] = None, connection_graph: Optional[nx.DiGraph] = None):
        """
        Initialize the AdjustableWorkflow.
        
        Args:
            agent_names: List of agent names to include in the network
            model_name: Optional model name to use (defaults to environment variable)
            connection_graph: Optional directed graph defining agent connections
        """
        # Initialize the connection graph if not provided
        self.connection_graph = connection_graph or self._create_default_connection_graph(agent_names)
        
        # Call the parent class constructor
        super().__init__(agent_names, model_name)
        
        # Log the connection graph
        self._log_connection_graph()
    
    def _create_default_connection_graph(self, agent_names: List[str]) -> nx.DiGraph:
        """
        Create a default fully connected graph if none is provided.
        
        Args:
            agent_names: List of agent names
            
        Returns:
            A directed graph with connections between all agents
        """
        graph = nx.DiGraph()
        
        # Add all agents as nodes
        for agent in agent_names:
            graph.add_node(agent)
        
        # Add edges between all agents (fully connected)
        for source in agent_names:
            for target in agent_names:
                if source != target:
                    graph.add_edge(source, target)
            
            # Add edge to END
            graph.add_edge(source, END)
        
        return graph
    
    def _log_connection_graph(self):
        """Log the current connection graph structure."""
        logger.info("Connection Graph Structure:", extra={"agent_name": "System"})
        
        for agent in self.agent_names:
            # Get all outgoing connections for this agent
            connections = list(self.connection_graph.successors(agent))
            logger.info(f"{agent} can route to: {', '.join(connections)}", extra={"agent_name": "System"})
    
    def update_connection_graph(self, new_graph: nx.DiGraph):
        """
        Update the connection graph at runtime.
        
        Args:
            new_graph: The new directed graph defining agent connections
        """
        # Validate the new graph
        self._validate_connection_graph(new_graph)
        
        # Update the connection graph
        self.connection_graph = new_graph
        
        # Log the updated graph
        logger.info("Connection graph updated", extra={"agent_name": "System"})
        self._log_connection_graph()
    
    def _validate_connection_graph(self, graph: nx.DiGraph):
        """
        Validate that the connection graph contains all agents and has valid connections.
        
        Args:
            graph: The graph to validate
            
        Raises:
            ValueError: If the graph is invalid
        """
        # Check that all agents are in the graph
        for agent in self.agent_names:
            if agent not in graph.nodes:
                raise ValueError(f"Agent '{agent}' is missing from the connection graph")
        
        # Check that all agents have at least one outgoing edge
        for agent in self.agent_names:
            if len(list(graph.successors(agent))) == 0:
                raise ValueError(f"Agent '{agent}' has no outgoing connections in the graph")
    
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
            
            # Count how many times this agent has been called
            agent_call_count = sum(1 for msg in messages if isinstance(msg, AIMessage) and msg.content.startswith(f"[{agent_name}]"))
            
            # Log that this agent is being called
            logger.info(f"Calling {agent_name} (call #{agent_call_count + 1})", extra={"agent_name": agent_name})
            
            # Get the allowed next agents from the connection graph
            allowed_next_agents = list(self.connection_graph.successors(agent_name))
            
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
                update={"messages": state["messages"] + [ai_message]}
            )
        
        return agent_node
    
    def _build_graph(self) -> StateGraph:
        """
        Build the graph with connections defined by the connection graph.
        
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
        
        # We don't need to add edges between agents since we're using Command objects
        # Each agent will decide where to go next using the Command return value
        # The connection graph is used to constrain which agents can be selected
        
        # Log the graph structure
        logger.info("Graph structure:", extra={"agent_name": "System"})
        logger.info(f"Agents: {', '.join(self.agent_names)}", extra={"agent_name": "System"})
        logger.info(f"Starting with: {self.agent_names[0]}", extra={"agent_name": "System"})
        logger.info("Using adjustable routing based on connection graph", extra={"agent_name": "System"})
        
        # Compile the graph
        return builder.compile()
    
    def add_connection(self, source: str, target: str):
        """
        Add a connection between two agents.
        
        Args:
            source: The source agent
            target: The target agent
            
        Raises:
            ValueError: If either agent is not in the workflow
        """
        if source not in self.agent_names:
            raise ValueError(f"Source agent '{source}' is not in the workflow")
        
        if target not in self.agent_names and target != END:
            raise ValueError(f"Target agent '{target}' is not in the workflow")
        
        # Add the edge to the connection graph
        self.connection_graph.add_edge(source, target)
        
        logger.info(f"Added connection: {source} → {target}", extra={"agent_name": "System"})
    
    def remove_connection(self, source: str, target: str):
        """
        Remove a connection between two agents.
        
        Args:
            source: The source agent
            target: The target agent
            
        Raises:
            ValueError: If either agent is not in the workflow or the connection doesn't exist
        """
        if source not in self.agent_names:
            raise ValueError(f"Source agent '{source}' is not in the workflow")
        
        if target not in self.agent_names and target != END:
            raise ValueError(f"Target agent '{target}' is not in the workflow")
        
        # Check if the edge exists
        if not self.connection_graph.has_edge(source, target):
            raise ValueError(f"Connection from '{source}' to '{target}' does not exist")
        
        # Remove the edge from the connection graph
        self.connection_graph.remove_edge(source, target)
        
        # Check if the source agent still has outgoing connections
        if len(list(self.connection_graph.successors(source))) == 0:
            # Add a connection to END if there are no other connections
            self.connection_graph.add_edge(source, END)
            logger.info(f"Added connection to END for {source} as it had no other outgoing connections", extra={"agent_name": "System"})
        
        logger.info(f"Removed connection: {source} → {target}", extra={"agent_name": "System"})
    
    def get_connections(self, agent_name: str) -> List[str]:
        """
        Get all outgoing connections for an agent.
        
        Args:
            agent_name: The agent to get connections for
            
        Returns:
            A list of agent names that this agent can route to
            
        Raises:
            ValueError: If the agent is not in the workflow
        """
        if agent_name not in self.agent_names:
            raise ValueError(f"Agent '{agent_name}' is not in the workflow")
        
        return list(self.connection_graph.successors(agent_name)) 