"""
This script provides detailed tests to verify that agents strictly follow
the connection graph in the AdjustableWorkflow implementation.
"""

import os
import re
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from config_workflow import ConfigWorkflow

# Load environment variables
load_dotenv()

def visualize_graph(graph, title="Connection Graph", output_file="connection_graph.png"):
    """
    Visualize the connection graph using matplotlib.
    
    Args:
        graph: The networkx graph to visualize
        title: The title of the plot
        output_file: The file to save the visualization to
    """
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color="lightblue")
    
    # Draw edges
    nx.draw_networkx_edges(graph, pos, width=1.5, arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight="bold")
    
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Graph visualization saved to {output_file}")

def extract_agent_flow(messages: List[Dict[str, Any]]) -> List[str]:
    """
    Extract the sequence of agents from the conversation history.
    
    Args:
        messages: The conversation history
        
    Returns:
        A list of agent names in the order they were called
    """
    agent_flow = []
    
    # Skip the first message (the query)
    for message in messages[1:]:
        if isinstance(message, AIMessage):
            # Extract agent name from the message content
            match = re.match(r'\[(.*?)\]:', message.content)
            if match:
                agent_name = match.group(1)
                agent_flow.append(agent_name)
    
    return agent_flow

def validate_agent_flow(agent_flow: List[str], connection_graph: nx.DiGraph) -> bool:
    """
    Validate that the agent flow follows the connection graph.
    
    Args:
        agent_flow: The sequence of agents in the conversation
        connection_graph: The connection graph defining allowed transitions
        
    Returns:
        True if the flow is valid, False otherwise
    """
    if not agent_flow:
        return True
    
    for i in range(len(agent_flow) - 1):
        source = agent_flow[i]
        target = agent_flow[i + 1]
        
        # Check if this transition is allowed in the connection graph
        if not connection_graph.has_edge(source, target):
            print(f"❌ Invalid transition: {source} -> {target}")
            return False
    
    return True

def test_linear_workflow():
    """
    Test a linear workflow where agents can only follow a specific path.
    
    This tests that agents strictly follow the connection graph constraints.
    """
    print("\n=== Testing Linear Workflow ===\n")
    
    # Create a linear connection graph: Researcher -> Critic -> Synthesizer -> END
    graph = nx.DiGraph()
    
    # Add nodes
    graph.add_node("Researcher")
    graph.add_node("Critic")
    graph.add_node("Synthesizer")
    
    # Add edges for linear flow only
    graph.add_edge("Researcher", "Critic")
    graph.add_edge("Critic", "Synthesizer")
    graph.add_edge("Synthesizer", "END")
    
    # Visualize the graph
    visualize_graph(graph, "Linear Connection Graph", "linear_connection_graph.png")
    
    # Create the workflow with the linear graph
    workflow = ConfigWorkflow(
        agent_names=["Researcher", "Critic", "Synthesizer"],
        connection_graph=graph
    )
    
    # Run the workflow
    query = "What are the key considerations for implementing a multi-agent system?"
    print(f"Query: {query}")
    
    result = workflow.run(query)
    
    # Extract the agent flow
    agent_flow = extract_agent_flow(result["messages"])
    print(f"Agent flow: {' -> '.join(agent_flow)}")
    
    # Validate the agent flow
    is_valid = validate_agent_flow(agent_flow, graph)
    print(f"Flow validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Print the conversation history
    print("\n=== Conversation History ===\n")
    for i, message in enumerate(result["messages"]):
        if i == 0:  # Skip the initial query
            continue
        print(f"{message.content}\n")
    
    return is_valid

def test_branching_workflow():
    """
    Test a branching workflow where some agents have multiple possible next agents.
    
    This tests that agents respect the connection graph when multiple options are available.
    """
    print("\n=== Testing Branching Workflow ===\n")
    
    # Create a branching connection graph:
    # Researcher -> Critic
    # Researcher -> Synthesizer
    # Critic -> Synthesizer
    # Synthesizer -> END
    graph = nx.DiGraph()
    
    # Add nodes
    graph.add_node("Researcher")
    graph.add_node("Critic")
    graph.add_node("Synthesizer")
    
    # Add edges for branching flow
    graph.add_edge("Researcher", "Critic")
    graph.add_edge("Researcher", "Synthesizer")
    graph.add_edge("Critic", "Synthesizer")
    graph.add_edge("Synthesizer", "END")
    
    # Visualize the graph
    visualize_graph(graph, "Branching Connection Graph", "branching_connection_graph.png")
    
    # Create the workflow with the branching graph
    workflow = ConfigWorkflow(
        agent_names=["Researcher", "Critic", "Synthesizer"],
        connection_graph=graph
    )
    
    # Run the workflow
    query = "What are the key considerations for implementing a multi-agent system?"
    print(f"Query: {query}")
    
    result = workflow.run(query)
    
    # Extract the agent flow
    agent_flow = extract_agent_flow(result["messages"])
    print(f"Agent flow: {' -> '.join(agent_flow)}")
    
    # Validate the agent flow
    is_valid = validate_agent_flow(agent_flow, graph)
    print(f"Flow validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Print the conversation history
    print("\n=== Conversation History ===\n")
    for i, message in enumerate(result["messages"]):
        if i == 0:  # Skip the initial query
            continue
        print(f"{message.content}\n")
    
    return is_valid

def test_cyclic_workflow():
    """
    Test a cyclic workflow where agents can loop back to previous agents.
    
    This tests that agents can handle cycles in the connection graph.
    """
    print("\n=== Testing Cyclic Workflow ===\n")
    
    # Create a cyclic connection graph:
    # Researcher -> Critic -> Synthesizer -> Researcher
    # All agents can also go to END
    graph = nx.DiGraph()
    
    # Add nodes
    graph.add_node("Researcher")
    graph.add_node("Critic")
    graph.add_node("Synthesizer")
    
    # Add edges for cyclic flow
    graph.add_edge("Researcher", "Critic")
    graph.add_edge("Critic", "Synthesizer")
    graph.add_edge("Synthesizer", "Researcher")
    
    # Add edges to END
    graph.add_edge("Researcher", "END")
    graph.add_edge("Critic", "END")
    graph.add_edge("Synthesizer", "END")
    
    # Visualize the graph
    visualize_graph(graph, "Cyclic Connection Graph", "cyclic_connection_graph.png")
    
    # Create the workflow with the cyclic graph
    workflow = ConfigWorkflow(
        agent_names=["Researcher", "Critic", "Synthesizer"],
        connection_graph=graph
    )
    
    # Run the workflow
    query = "What are the key considerations for implementing a multi-agent system?"
    print(f"Query: {query}")
    
    result = workflow.run(query)
    
    # Extract the agent flow
    agent_flow = extract_agent_flow(result["messages"])
    print(f"Agent flow: {' -> '.join(agent_flow)}")
    
    # Validate the agent flow
    is_valid = validate_agent_flow(agent_flow, graph)
    print(f"Flow validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Print the conversation history
    print("\n=== Conversation History ===\n")
    for i, message in enumerate(result["messages"]):
        if i == 0:  # Skip the initial query
            continue
        print(f"{message.content}\n")
    
    return is_valid

def test_restricted_workflow():
    """
    Test a restricted workflow where some agents have very limited options.
    
    This tests that agents respect strict constraints in the connection graph.
    """
    print("\n=== Testing Restricted Workflow ===\n")
    
    # Create a restricted connection graph:
    # Researcher -> Critic
    # Critic -> END
    # Synthesizer -> Researcher
    graph = nx.DiGraph()
    
    # Add nodes
    graph.add_node("Researcher")
    graph.add_node("Critic")
    graph.add_node("Synthesizer")
    
    # Add edges for restricted flow
    graph.add_edge("Researcher", "Critic")
    graph.add_edge("Critic", "END")
    graph.add_edge("Synthesizer", "Researcher")
    
    # Visualize the graph
    visualize_graph(graph, "Restricted Connection Graph", "restricted_connection_graph.png")
    
    # Create the workflow with the restricted graph
    workflow = ConfigWorkflow(
        agent_names=["Researcher", "Critic", "Synthesizer"],
        connection_graph=graph
    )
    
    # Run the workflow
    query = "What are the key considerations for implementing a multi-agent system?"
    print(f"Query: {query}")
    
    result = workflow.run(query)
    
    # Extract the agent flow
    agent_flow = extract_agent_flow(result["messages"])
    print(f"Agent flow: {' -> '.join(agent_flow)}")
    
    # Validate the agent flow
    is_valid = validate_agent_flow(agent_flow, graph)
    print(f"Flow validity: {'✅ Valid' if is_valid else '❌ Invalid'}")
    
    # Print the conversation history
    print("\n=== Conversation History ===\n")
    for i, message in enumerate(result["messages"]):
        if i == 0:  # Skip the initial query
            continue
        print(f"{message.content}\n")
    
    return is_valid

if __name__ == "__main__":
    # Run all tests
    try:
        results = []
        
        # Test linear workflow
        results.append(("Linear Workflow", test_linear_workflow()))
        
        # Test branching workflow
        results.append(("Branching Workflow", test_branching_workflow()))
        
        # Test cyclic workflow
        results.append(("Cyclic Workflow", test_cyclic_workflow()))
        
        # Test restricted workflow
        results.append(("Restricted Workflow", test_restricted_workflow()))
        
        
        # Print summary of results
        print("\n=== Test Results Summary ===\n")
        all_passed = True
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            if not result:
                all_passed = False
            print(f"{test_name}: {status}")
        
        print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc() 