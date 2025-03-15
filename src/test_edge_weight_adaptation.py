"""
Test script for edge weight adaptation in the AdjustableWorkflow.

This script focuses specifically on testing the ability of the AdjustableWorkflow
to adapt edge weights from another graph, demonstrating how the workflow can
dynamically adjust its routing behavior based on external graph configurations.
"""

import os
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple, Counter
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END

from src.adjustable_workflow import AdjustableWorkflow
from src.connection_manager import ConnectionManager

def extract_agent_flow(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Extract the sequence of agents from a list of messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of agent names in the order they were called
    """
    agent_flow = []
    
    for msg in messages:
        if msg.type == "ai" and msg.content.startswith("["):
            # Extract agent name from message content
            agent_name = msg.content.split("]")[0].strip("[")
            agent_flow.append(agent_name)
    
    return agent_flow

def create_weighted_graph(agent_names: List[str], edge_weights: Dict[Tuple[str, str], float]) -> nx.DiGraph:
    """
    Create a directed graph with weighted edges.
    
    Args:
        agent_names: List of agent names to include as nodes
        edge_weights: Dictionary mapping (source, target) tuples to edge weights
        
    Returns:
        A directed graph with weighted edges
    """
    graph = nx.DiGraph()
    
    # Add nodes
    for agent in agent_names:
        graph.add_node(agent)
    graph.add_node(END)
    
    # Add edges with weights
    for (source, target), weight in edge_weights.items():
        graph.add_edge(source, target, weight=weight)
    
    return graph

def visualize_graph_comparison(graph1: nx.DiGraph, graph2: nx.DiGraph, title1: str, title2: str, output_file: str):
    """
    Create a side-by-side visualization of two graphs for comparison.
    
    Args:
        graph1: First graph to visualize
        graph2: Second graph to visualize
        title1: Title for the first graph
        title2: Title for the second graph
        output_file: File to save the visualization to
        
    Returns:
        Path to the saved visualization
    """
    plt.figure(figsize=(20, 10))
    
    # Use the same layout for both graphs for better comparison
    pos = nx.spring_layout(graph1)
    
    # Plot the first graph
    plt.subplot(1, 2, 1)
    nx.draw_networkx_nodes(graph1, pos, node_size=700, node_color="lightblue")
    nx.draw_networkx_edges(graph1, pos, width=1.5, arrowsize=20)
    nx.draw_networkx_labels(graph1, pos, font_size=12, font_weight="bold")
    
    # Draw edge weights for the first graph
    edge_labels1 = {(source, target): f"{data.get('weight', 1.0):.2f}" 
                   for source, target, data in graph1.edges(data=True)}
    nx.draw_networkx_edge_labels(graph1, pos, edge_labels=edge_labels1, font_size=10)
    
    plt.title(title1)
    plt.axis("off")
    
    # Plot the second graph
    plt.subplot(1, 2, 2)
    nx.draw_networkx_nodes(graph2, pos, node_size=700, node_color="lightgreen")
    nx.draw_networkx_edges(graph2, pos, width=1.5, arrowsize=20)
    nx.draw_networkx_labels(graph2, pos, font_size=12, font_weight="bold")
    
    # Draw edge weights for the second graph
    edge_labels2 = {(source, target): f"{data.get('weight', 1.0):.2f}" 
                   for source, target, data in graph2.edges(data=True)}
    nx.draw_networkx_edge_labels(graph2, pos, edge_labels=edge_labels2, font_size=10)
    
    plt.title(title2)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return output_file

def test_basic_weight_adaptation():
    """
    Test basic adaptation of edge weights from another graph.
    
    This test demonstrates how the workflow can adapt edge weights from
    another graph with a simple configuration.
    """
    print("\n=== Testing Basic Edge Weight Adaptation ===")
    
    # Define agent names
    agent_names = ["A", "B", "C", "D"]
    
    # Create an initial connection manager with equal weights
    connection_manager = ConnectionManager(agent_names, default_weight=1.0)
    
    # Create the workflow with the initial connection manager
    workflow = AdjustableWorkflow(agent_names, connection_manager=connection_manager)
    
    # Create a new graph with custom weights
    edge_weights = {
        ("A", "B"): 0.8,
        ("A", "C"): 0.2,
        ("B", "C"): 0.6,
        ("B", "D"): 0.4,
        ("C", "D"): 0.9,
        ("C", END): 0.1,
        ("D", END): 1.0
    }
    new_graph = create_weighted_graph(agent_names, edge_weights)
    
    # Visualize the initial and new graphs side by side
    comparison_path = visualize_graph_comparison(
        workflow.connection_manager.graph,
        new_graph,
        "Initial Graph (Equal Weights)",
        "New Graph (Custom Weights)",
        "basic_weight_adaptation_before.png"
    )
    print(f"Graph comparison visualization saved to: {comparison_path}")
    
    # Update the workflow with the new edge weights
    workflow.update_weights_from_graph(new_graph)
    
    # Visualize the updated graph
    updated_graph_path = workflow.visualize_connections(
        title="Updated Graph (After Weight Adaptation)",
        output_file="basic_weight_adaptation_after.png"
    )
    print(f"Updated graph visualization saved to: {updated_graph_path}")
    
    # Verify that the weights were updated correctly
    all_weights_correct = True
    for (source, target), expected_weight in edge_weights.items():
        if workflow.connection_manager.can_route_to(source, target):
            actual_weight = workflow.get_edge_weight(source, target)
            if abs(actual_weight - expected_weight) > 0.01:  # Allow for small floating-point differences
                print(f"Weight mismatch for {source} → {target}: expected {expected_weight}, got {actual_weight}")
                all_weights_correct = False
    
    print(f"All weights updated correctly: {all_weights_correct}")
    
    return all_weights_correct

def test_partial_graph_adaptation():
    """
    Test adaptation of edge weights from a partial graph.
    
    This test demonstrates how the workflow adapts weights from a graph
    that only contains a subset of the edges in the workflow graph.
    """
    print("\n=== Testing Partial Graph Adaptation ===")
    
    # Define agent names
    agent_names = ["X", "Y", "Z", "W"]
    
    # Create an initial connection manager with specific connections and weights
    connection_manager = ConnectionManager(agent_names, fully_connected=False)
    
    # Set up initial connections with weights
    initial_weights = {
        ("X", "Y"): 0.5,
        ("X", "Z"): 0.5,
        ("Y", "Z"): 0.7,
        ("Y", "W"): 0.3,
        ("Z", "W"): 0.8,
        ("Z", END): 0.2,
        ("W", END): 1.0
    }
    
    for (source, target), weight in initial_weights.items():
        connection_manager.add_connection(source, target, weight)
    
    # Create the workflow with the initial connection manager
    workflow = AdjustableWorkflow(agent_names, connection_manager=connection_manager)
    
    # Create a partial graph with only some of the edges
    partial_weights = {
        ("X", "Y"): 0.2,  # Changed
        ("X", "Z"): 0.8,  # Changed
        ("Y", "Z"): 0.7,  # Same
        # ("Y", "W") is missing
        ("Z", "W"): 0.9,  # Changed
        # ("Z", END) is missing
        ("W", END): 0.5   # Changed
    }
    partial_graph = create_weighted_graph(agent_names, partial_weights)
    
    # Visualize the initial and partial graphs side by side
    comparison_path = visualize_graph_comparison(
        workflow.connection_manager.graph,
        partial_graph,
        "Initial Graph (Complete)",
        "Partial Graph (Some Edges)",
        "partial_weight_adaptation_before.png"
    )
    print(f"Graph comparison visualization saved to: {comparison_path}")
    
    # Update the workflow with the partial graph weights
    workflow.update_weights_from_graph(partial_graph)
    
    # Visualize the updated graph
    updated_graph_path = workflow.visualize_connections(
        title="Updated Graph (After Partial Adaptation)",
        output_file="partial_weight_adaptation_after.png"
    )
    print(f"Updated graph visualization saved to: {updated_graph_path}")
    
    # Verify that only the weights in the partial graph were updated
    all_weights_correct = True
    
    # Check weights that should have been updated
    for (source, target), expected_weight in partial_weights.items():
        actual_weight = workflow.get_edge_weight(source, target)
        if abs(actual_weight - expected_weight) > 0.01:
            print(f"Weight mismatch for {source} → {target}: expected {expected_weight}, got {actual_weight}")
            all_weights_correct = False
    
    # Check weights that should remain unchanged
    unchanged_edges = [("Y", "W"), ("Z", END)]
    for source, target in unchanged_edges:
        expected_weight = initial_weights.get((source, target))
        actual_weight = workflow.get_edge_weight(source, target)
        if abs(actual_weight - expected_weight) > 0.01:
            print(f"Weight should be unchanged for {source} → {target}: expected {expected_weight}, got {actual_weight}")
            all_weights_correct = False
    
    print(f"All weights correctly adapted: {all_weights_correct}")
    
    return all_weights_correct
def test_complex_workflow_adaptation():
    """
    Test adaptation of edge weights in a complex workflow.
    
    This test demonstrates how the workflow can adapt edge weights in a more
    complex configuration with multiple paths and feedback loops.
    """
    print("\n=== Testing Complex Workflow Adaptation ===")
    
    # Define agent names
    agent_names = ["Start", "Process", "Validate", "Refine", "Finalize"]
    
    # Create a connection manager with a complex workflow
    connection_manager = ConnectionManager(agent_names, fully_connected=False)
    
    # Set up initial connections with weights
    # This creates a workflow with feedback loops
    initial_weights = {
        ("Start", "Process"): 1.0,
        ("Process", "Validate"): 0.7,
        ("Process", "Refine"): 0.3,
        ("Validate", "Refine"): 0.4,
        ("Validate", "Finalize"): 0.6,
        ("Refine", "Process"): 0.5,  # Feedback loop
        ("Refine", "Validate"): 0.5,
        ("Finalize", END): 1.0
    }
    
    for (source, target), weight in initial_weights.items():
        connection_manager.add_connection(source, target, weight)
    
    # Create the workflow with the initial connection manager
    workflow = AdjustableWorkflow(agent_names, connection_manager=connection_manager)
    
    # Visualize the initial complex workflow
    initial_graph_path = workflow.visualize_connections(
        title="Initial Complex Workflow",
        output_file="complex_workflow_initial.png"
    )
    print(f"Initial complex workflow visualization saved to: {initial_graph_path}")
    
    # Create a new graph with modified weights
    # This changes the routing preferences in the workflow
    modified_weights = {
        ("Start", "Process"): 1.0,
        ("Process", "Validate"): 0.3,  # Decreased
        ("Process", "Refine"): 0.7,    # Increased
        ("Validate", "Refine"): 0.2,   # Decreased
        ("Validate", "Finalize"): 0.8, # Increased
        ("Refine", "Process"): 0.8,    # Increased feedback
        ("Refine", "Validate"): 0.2,   # Decreased
        ("Finalize", END): 1.0
    }
    modified_graph = create_weighted_graph(agent_names, modified_weights)
    
    # Update the workflow with the modified weights
    workflow.update_weights_from_graph(modified_graph)
    
    # Visualize the updated workflow
    updated_graph_path = workflow.visualize_connections(
        title="Updated Complex Workflow",
        output_file="complex_workflow_updated.png"
    )
    print(f"Updated complex workflow visualization saved to: {updated_graph_path}")
    
    # Verify that the weights were updated correctly
    all_weights_correct = True
    for (source, target), expected_weight in modified_weights.items():
        actual_weight = workflow.get_edge_weight(source, target)
        if abs(actual_weight - expected_weight) > 0.01:
            print(f"Weight mismatch for {source} → {target}: expected {expected_weight}, got {actual_weight}")
            all_weights_correct = False
    
    print(f"All weights updated correctly: {all_weights_correct}")
    
    # Run a simulation to observe the effect of the weight changes
    print("\nRunning a simulation with the updated weights...")
    
    # Create a human message to start the workflow
    human_message = HumanMessage(content="Process this request through the complex workflow.")
    
    # Run the workflow
    result = workflow.invoke({"messages": [human_message]})
    
    # Extract the agent flow
    agent_flow = extract_agent_flow(result["messages"])
    print(f"Agent flow: {' -> '.join(agent_flow)}")
    
    # Check if the simulation completed
    simulation_completed = "Finalize" in agent_flow
    print(f"Simulation completed: {simulation_completed}")
    
    return all_weights_correct and simulation_completed

def run_all_tests():
    """Run all edge weight adaptation tests."""
    tests = [
        test_basic_weight_adaptation,
        test_partial_graph_adaptation,
        test_complex_workflow_adaptation
    ]
    
    results = []
    
    for test in tests:
        try:
            print(f"\n{'='*80}\nRunning {test.__name__}\n{'='*80}")
            result = test()
            results.append(result)
            print(f"\nTest {test.__name__}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"Error in {test.__name__}: {e}")
            results.append(False)
    
    # Print summary
    print("\n=== Test Summary ===")
    for i, test in enumerate(tests):
        print(f"{test.__name__}: {'PASS' if results[i] else 'FAIL'}")
    
    return all(results)

if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    if success:
        print("\nAll edge weight adaptation tests passed!")
    else:
        print("\nSome edge weight adaptation tests failed. Check the output for details.") 