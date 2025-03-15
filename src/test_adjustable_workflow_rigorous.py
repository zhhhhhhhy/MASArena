"""
Rigorous test script for the AdjustableWorkflow implementation.

This script demonstrates the dynamic modification of the workflow during execution,
showing how the connection graph can be updated while the workflow is running.
"""

import os
import time
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Any, Set, Optional, Tuple
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

def validate_agent_transitions(agent_flow: List[str], allowed_transitions: Dict[str, Set[str]]) -> bool:
    """
    Validate that all agent transitions follow the allowed transitions.
    
    Args:
        agent_flow: List of agent names in the order they were called
        allowed_transitions: Dictionary mapping agent names to sets of allowed next agents
        
    Returns:
        True if all transitions are valid, False otherwise
    """
    for i in range(len(agent_flow) - 1):
        current_agent = agent_flow[i]
        next_agent = agent_flow[i + 1]
        
        if next_agent not in allowed_transitions.get(current_agent, set()):
            print(f"Invalid transition: {current_agent} -> {next_agent}")
            print(f"Allowed transitions for {current_agent}: {allowed_transitions.get(current_agent, set())}")
            return False
    
    return True

def test_dynamic_modification_during_execution():
    """
    Test dynamic modification of the workflow during execution.
    
    This test demonstrates how the connection graph can be modified
    while the workflow is running, affecting future agent routing.
    """
    print("\n=== Testing Dynamic Modification During Execution ===")
    
    # Define agent names
    agent_names = ["Coordinator", "Researcher", "Writer", "Editor", "Reviewer"]
    
    # Create an initial connection manager with a specific flow
    connection_manager = ConnectionManager(agent_names)
    connection_manager.set_connections("Coordinator", ["Researcher"])
    connection_manager.set_connections("Researcher", ["Writer"])
    connection_manager.set_connections("Writer", ["Editor"])
    connection_manager.set_connections("Editor", ["Reviewer"])
    connection_manager.set_connections("Reviewer", [END])
    
    # Create the workflow with the initial connection manager
    workflow = AdjustableWorkflow(agent_names, connection_manager=connection_manager)
    
    # Define the initial allowed transitions
    initial_allowed_transitions = {
        "Coordinator": {"Researcher"},
        "Researcher": {"Writer"},
        "Writer": {"Editor"},
        "Editor": {"Reviewer"},
        "Reviewer": {END}
    }
    
    # Create a human message to start the workflow
    human_message = HumanMessage(content="Write a comprehensive report on renewable energy technologies.")
    
    # Create a state to track the workflow execution
    state = {"messages": [human_message]}
    
    # Run the workflow for a few steps
    print("Running workflow with initial configuration...")
    
    # Step 1: Coordinator -> Researcher
    state = workflow.graph.invoke(state)
    
    # Extract the current agent flow
    current_agent_flow = extract_agent_flow(state["messages"])
    current_flow_index = len(current_agent_flow) - 1
    print(f"Current agent flow: {' -> '.join(current_agent_flow)}")
    
    # Validate the transitions so far
    is_valid_initial = validate_agent_transitions(current_agent_flow, initial_allowed_transitions)
    print(f"Valid transitions (initial): {is_valid_initial}")
    
    # Now, dynamically modify the workflow connections
    print("\nDynamically modifying workflow connections...")
    
    # Update the connection manager to skip the Editor
    workflow.set_connections("Writer", ["Reviewer"])
    
    # Define the modified allowed transitions
    modified_allowed_transitions = {
        "Coordinator": {"Researcher"},
        "Researcher": {"Writer"},
        "Writer": {"Reviewer"},  # Writer now goes directly to Reviewer
        "Editor": {"Reviewer"},
        "Reviewer": {END}
    }
    
    # Continue running the workflow with the modified connections
    print("Continuing workflow with modified configuration...")
    
       
    # Run the next step
    state = workflow.graph.invoke(state)
    
    # Update the agent flow
    current_agent_flow = extract_agent_flow(state["messages"])
    print(f"Updated agent flow: {' -> '.join(current_agent_flow)}")
    
    # Extract the final agent flow
    final_agent_flow = extract_agent_flow(state["messages"])[current_flow_index+1:]
    print(f"Final agent flow: {' -> '.join(final_agent_flow)}")
    
    # Validate the transitions with the modified configuration
    is_valid_modified = validate_agent_transitions(final_agent_flow, modified_allowed_transitions)
    print(f"Valid transitions (modified): {is_valid_modified}")
    
    # Check if the Editor was skipped as expected
    writer_index = -1
    reviewer_index = -1
    
    for i, agent in enumerate(final_agent_flow):
        if agent == "Writer":
            writer_index = i
        elif agent == "Reviewer" and writer_index != -1 and reviewer_index == -1:
            reviewer_index = i
    
    editor_skipped = writer_index != -1 and reviewer_index != -1 and reviewer_index == writer_index + 1
    print(f"Editor was skipped: {editor_skipped}")
    
    return is_valid_initial and is_valid_modified and editor_skipped

def test_conditional_routing_based_on_content():
    """
    Test conditional routing based on message content.
    
    This test demonstrates how the workflow can be dynamically modified
    based on the content of messages, enabling content-aware routing.
    """
    print("\n=== Testing Conditional Routing Based on Content ===")
    
    # Define agent names
    agent_names = ["Dispatcher", "TechnicalExpert", "BusinessAnalyst", "CustomerSupport"]
    
    # Create a fully connected initial configuration
    connection_manager = ConnectionManager(agent_names)
    connection_manager.reset_to_fully_connected()
    
    # Create the workflow
    workflow = AdjustableWorkflow(agent_names, connection_manager=connection_manager)
    
    # Define a function to analyze message content and update routing
    def update_routing_based_on_content(query: str, workflow: AdjustableWorkflow):
        """
        Analyze message content and update routing accordingly.
        
        Args:
            query: The query text to analyze
            workflow: The AdjustableWorkflow instance to update
        """
        # Convert to lowercase for case-insensitive matching
        content = query.lower()
        
        # Check for technical keywords
        technical_keywords = ["code", "programming", "technical", "bug", "error", "software"]
        business_keywords = ["cost", "budget", "strategy", "market", "business", "profit"]
        support_keywords = ["help", "support", "assistance", "customer", "service"]
        
        # Reset connections for the Dispatcher
        workflow.set_connections("Dispatcher", [])
        
        # Add connections based on content
        if any(keyword in content for keyword in technical_keywords):
            workflow.add_connection("Dispatcher", "TechnicalExpert")
            print("Detected technical content, routing to TechnicalExpert")
        
        if any(keyword in content for keyword in business_keywords):
            workflow.add_connection("Dispatcher", "BusinessAnalyst")
            print("Detected business content, routing to BusinessAnalyst")
        
        if any(keyword in content for keyword in support_keywords):
            workflow.add_connection("Dispatcher", "CustomerSupport")
            print("Detected support content, routing to CustomerSupport")
        
        # If no specific routing was added, connect to all agents
        if not workflow.get_connections("Dispatcher"):
            workflow.set_connections("Dispatcher", ["TechnicalExpert", "BusinessAnalyst", "CustomerSupport"])
            print("No specific content detected, routing to all agents")
    
    # Test with different queries
    test_queries = [
        "I need help with a technical bug in my code",
        "What's the business strategy for expanding into new markets?",
        "Can you provide customer support for my recent purchase?"
    ]
    
    results = []
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        
        # Create a human message
        human_message = HumanMessage(content=query)
        
        # Reset the workflow to fully connected
        workflow.reset_to_fully_connected()
        
        # Update routing based on the query content
        update_routing_based_on_content(query, workflow)
        
        # Run the workflow
        result = workflow.invoke({"messages": [human_message]})
        
        # Extract the agent flow
        agent_flow = extract_agent_flow(result["messages"])
        print(f"Agent flow: {' -> '.join(agent_flow)}")
        
        # Check if the routing was appropriate
        if "technical" in query.lower() and "TechnicalExpert" in agent_flow:
            print("✓ Technical query correctly routed to TechnicalExpert")
            results.append(True)
        elif "business" in query.lower() and "BusinessAnalyst" in agent_flow:
            print("✓ Business query correctly routed to BusinessAnalyst")
            results.append(True)
        elif "customer support" in query.lower() and "CustomerSupport" in agent_flow:
            print("✓ Support query correctly routed to CustomerSupport")
            results.append(True)
        else:
            print("✗ Query not routed as expected")
            results.append(False)
    
    return all(results)

def test_runtime_graph_visualization():
    """
    Test runtime visualization of the connection graph.
    
    This test demonstrates how the connection graph can be visualized
    at runtime, providing insights into the current routing configuration.
    """
    print("\n=== Testing Runtime Graph Visualization ===")
    
    # Define agent names
    agent_names = ["Orchestrator", "Analyzer", "Processor", "Finalizer"]
    
    # Create a connection manager
    connection_manager = ConnectionManager(agent_names)
    connection_manager.set_connections("Orchestrator", ["Analyzer", "Processor"])
    connection_manager.set_connections("Analyzer", ["Processor"])
    connection_manager.set_connections("Processor", ["Finalizer"])
    connection_manager.set_connections("Finalizer", [END])
    
    # Create the workflow
    workflow = AdjustableWorkflow(agent_names, connection_manager=connection_manager)
    
    # Visualize the initial graph
    initial_graph_path = workflow.visualize_connections(
        title="Initial Connection Graph",
        output_file="initial_connection_graph.png"
    )
    print(f"Initial graph visualization saved to: {initial_graph_path}")
    
    # Modify the graph
    workflow.add_connection("Analyzer", "Finalizer")
    workflow.add_connection("Orchestrator", "Finalizer")
    
    # Visualize the modified graph
    modified_graph_path = workflow.visualize_connections(
        title="Modified Connection Graph",
        output_file="modified_connection_graph.png"
    )
    print(f"Modified graph visualization saved to: {modified_graph_path}")
    
    # Check if the visualization files were created
    initial_exists = os.path.exists(initial_graph_path)
    modified_exists = os.path.exists(modified_graph_path)
    
    print(f"Initial graph visualization file exists: {initial_exists}")
    print(f"Modified graph visualization file exists: {modified_exists}")
    
    return initial_exists and modified_exists

def run_all_tests():
    """Run all rigorous tests."""
    tests = [
        test_dynamic_modification_during_execution,
        test_conditional_routing_based_on_content,
        test_runtime_graph_visualization
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
        print("\nAll rigorous tests passed!")
    else:
        print("\nSome rigorous tests failed. Check the output for details.") 