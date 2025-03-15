"""
ConnectionManager: Manages the connections between agents in a multi-agent system.

This class provides methods to query and update the connection graph,
allowing for dynamic adjustment of agent routing at runtime.
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from langgraph.graph import END

class ConnectionManager:
    """
    Manages the connections between agents in a multi-agent system.
    
    This class maintains a directed graph representing the allowed connections
    between agents and provides methods to query and update this graph.
    """
    
    def __init__(self, agent_names: List[str], fully_connected: bool = True, default_weight: float = 1.0):
        """
        Initialize the ConnectionManager.
        
        Args:
            agent_names: List of agent names to include in the graph
            fully_connected: Whether to create a fully connected graph initially
            default_weight: Default weight for edges when not specified
        """
        self.agent_names = agent_names
        self.graph = nx.DiGraph()
        self.default_weight = default_weight
        
        # Add all agents as nodes
        for agent in agent_names:
            self.graph.add_node(agent)
        
        # Add edges based on the fully_connected parameter
        if fully_connected:
            self._create_fully_connected_graph()
        
        # Add END as a special node
        self.graph.add_node(END)
    
    def _create_fully_connected_graph(self):
        """Create a fully connected graph where each agent can route to any other agent."""
        for source in self.agent_names:
            for target in self.agent_names:
                if source != target:
                    self.graph.add_edge(source, target, weight=self.default_weight)
            
            # Add edge to END
            self.graph.add_edge(source, END, weight=self.default_weight)
    
    def get_allowed_next_agents(self, agent_name: str) -> List[str]:
        """
        Get the list of agents that the specified agent can route to.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            A list of agent names that this agent can route to
        """
        if agent_name not in self.graph:
            raise ValueError(f"Agent '{agent_name}' is not in the graph")
        
        return list(self.graph.successors(agent_name))
    
    def get_weighted_next_agents(self, agent_name: str) -> Dict[str, float]:
        """
        Get the list of agents that the specified agent can route to, with their weights.
        
        Args:
            agent_name: The name of the agent
            
        Returns:
            A dictionary mapping agent names to their edge weights
        """
        if agent_name not in self.graph:
            raise ValueError(f"Agent '{agent_name}' is not in the graph")
        
        return {target: data.get('weight', self.default_weight) 
                for target, data in self.graph[agent_name].items()}
    
    def can_route_to(self, source: str, target: str) -> bool:
        """
        Check if the source agent can route to the target agent.
        
        Args:
            source: The source agent name
            target: The target agent name
            
        Returns:
            True if the source can route to the target, False otherwise
        """
        return self.graph.has_edge(source, target)
    
    def add_connection(self, source: str, target: str, weight: Optional[float] = None):
        """
        Add a connection from the source agent to the target agent.
        
        Args:
            source: The source agent name
            target: The target agent name
            weight: Optional weight for the edge (defaults to self.default_weight)
        """
        if source not in self.agent_names:
            raise ValueError(f"Source agent '{source}' is not in the graph")
        
        if target not in self.agent_names and target != END:
            raise ValueError(f"Target agent '{target}' is not in the graph")
        
        self.graph.add_edge(source, target, weight=weight if weight is not None else self.default_weight)
    
    def remove_connection(self, source: str, target: str):
        """
        Remove a connection from the source agent to the target agent.
        
        Args:
            source: The source agent name
            target: The target agent name
        """
        if source not in self.agent_names:
            raise ValueError(f"Source agent '{source}' is not in the graph")
        
        if target not in self.agent_names and target != END:
            raise ValueError(f"Target agent '{target}' is not in the graph")
        
        if not self.graph.has_edge(source, target):
            raise ValueError(f"Connection from '{source}' to '{target}' does not exist")
        
        self.graph.remove_edge(source, target)
        
        # Ensure the agent has at least one outgoing connection
        if len(list(self.graph.successors(source))) == 0:
            self.graph.add_edge(source, END, weight=self.default_weight)
    
    def set_connections(self, agent_name: str, targets: List[str], weights: Optional[List[float]] = None):
        """
        Set the connections for an agent, replacing any existing connections.
        
        Args:
            agent_name: The agent name
            targets: List of target agent names
            weights: Optional list of weights for each target (must match targets length)
        """
        if agent_name not in self.agent_names:
            raise ValueError(f"Agent '{agent_name}' is not in the graph")
        
        # Validate targets
        for target in targets:
            if target not in self.agent_names and target != END:
                raise ValueError(f"Target agent '{target}' is not in the graph")
        
        # Validate weights if provided
        if weights is not None and len(weights) != len(targets):
            raise ValueError("Length of weights must match length of targets")
        
        # Remove all existing connections
        for target in list(self.graph.successors(agent_name)):
            self.graph.remove_edge(agent_name, target)
        
        # Add new connections with weights
        for i, target in enumerate(targets):
            weight = weights[i] if weights is not None else self.default_weight
            self.graph.add_edge(agent_name, target, weight=weight)
        
        # Ensure the agent has at least one outgoing connection
        if len(targets) == 0:
            self.graph.add_edge(agent_name, END, weight=self.default_weight)
    
    def set_edge_weight(self, source: str, target: str, weight: float):
        """
        Set the weight of an existing edge.
        
        Args:
            source: The source agent name
            target: The target agent name
            weight: The new weight for the edge
        """
        if not self.graph.has_edge(source, target):
            raise ValueError(f"Connection from '{source}' to '{target}' does not exist")
        
        self.graph[source][target]['weight'] = weight
    
    def get_edge_weight(self, source: str, target: str) -> float:
        """
        Get the weight of an existing edge.
        
        Args:
            source: The source agent name
            target: The target agent name
            
        Returns:
            The weight of the edge
        """
        if not self.graph.has_edge(source, target):
            raise ValueError(f"Connection from '{source}' to '{target}' does not exist")
        
        return self.graph[source][target].get('weight', self.default_weight)
    
    def update_weights_from_graph(self, other_graph: nx.DiGraph):
        """
        Update edge weights from another graph.
        
        Args:
            other_graph: Another directed graph with edge weights
        """
        for source, target, data in other_graph.edges(data=True):
            if self.graph.has_edge(source, target):
                weight = data.get('weight', self.default_weight)
                self.graph[source][target]['weight'] = weight
    
    def reset_to_fully_connected(self):
        """Reset the graph to a fully connected configuration."""
        # Remove all edges
        self.graph.remove_edges_from(list(self.graph.edges()))
        
        # Create a fully connected graph
        self._create_fully_connected_graph()
    
    def visualize(self, title: str = "Agent Connection Graph", output_file: str = "connection_graph.png", show_weights: bool = True):
        """
        Visualize the connection graph and save it to a file.
        
        Args:
            title: The title of the plot
            output_file: The file to save the visualization to
            show_weights: Whether to display edge weights
        """
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color="lightblue")
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=1.5, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight="bold")
        
        # Draw edge weights if requested
        if show_weights:
            edge_labels = {(source, target): f"{data.get('weight', self.default_weight):.2f}" 
                          for source, target, data in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=10)
        
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        
        return output_file
    
    def to_dict(self) -> Dict[str, List[Union[str, Tuple[str, float]]]]:
        """
        Convert the connection graph to a dictionary.
        
        Returns:
            A dictionary mapping agent names to lists of target agent names with weights
        """
        result = {}
        for agent in self.agent_names:
            result[agent] = [(target, data.get('weight', self.default_weight)) 
                            for target, data in self.graph[agent].items()]
        return result
    
    @classmethod
    def from_dict(cls, agent_names: List[str], connections: Dict[str, List[Union[str, Tuple[str, float]]]]):
        """
        Create a ConnectionManager from a dictionary.
        
        Args:
            agent_names: List of agent names
            connections: Dictionary mapping agent names to lists of target agent names with weights
            
        Returns:
            A new ConnectionManager instance
        """
        manager = cls(agent_names, fully_connected=False)
        
        for source, targets in connections.items():
            for target_info in targets:
                if isinstance(target_info, tuple):
                    target, weight = target_info
                    manager.add_connection(source, target, weight)
                else:
                    manager.add_connection(source, target_info)
        
        return manager 