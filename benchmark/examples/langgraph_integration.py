"""
LangGraph Multi-Agent System Metrics Integration Example.

This example demonstrates how to integrate the metrics framework with a LangGraph-based
multi-agent system to collect and analyze system-level performance metrics.
"""

import os
import time
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import dotenv
dotenv.load_dotenv()

# Import our metrics framework
from benchmark.src.metrics import (
    MetricsRegistry,
    MetricsCollectionConfig,
    SystemMetricsCollector,
    AgentMetricsCollector, 
    InterAgentMetricsCollector
)

# Import config classes
from benchmark.src.metrics.system_metrics import SystemMetricsConfig
from benchmark.src.metrics.agent_metrics import AgentMetricsConfig
from benchmark.src.metrics.inter_agent_metrics import InterAgentMetricsConfig


# Initialize the metrics collectors
def setup_metrics(output_dir: str = "metrics_output") -> MetricsRegistry:
    """Set up metrics collection for a LangGraph-based system."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the metrics registry
    registry = MetricsRegistry()
    
    # Configure and register system metrics collector
    system_config = SystemMetricsConfig(
        sampling_interval_ms=5000,            # Sample every 5 seconds
        metrics_storage_path=os.path.join(output_dir, "system_metrics.json"),
        monitor_gpu=True,                     # Enable GPU monitoring if available
        # Queue-based processing configuration
        metrics_queue_size=5000,              # Size of metrics processing queue
        metrics_batch_size=50,                # Process metrics in batches of 50
        metrics_flush_interval_ms=100,        # Flush metrics every 100ms
        sampling_rate=0.8                     # Sample 80% of metrics (reduce overhead)
    )
    system_collector = SystemMetricsCollector(system_config)
    registry.register_collector("system", system_collector)
    
    # Configure and register agent metrics collector
    agent_config = AgentMetricsConfig(
        sampling_interval_ms=1000,            # Sample every 1 second
        metrics_storage_path=os.path.join(output_dir, "agent_metrics.json"),
        track_token_usage=True,
        track_model_latency=True,
        track_memory_operations=True,
        # Queue-based processing configuration
        metrics_queue_size=10000,             # Larger queue for agent metrics
        metrics_batch_size=100,               # Process in larger batches
        metrics_flush_interval_ms=250,        # Flush less frequently
        sampling_rate=1.0                     # Collect all agent metrics (important data)
    )
    agent_collector = AgentMetricsCollector(agent_config)
    registry.register_collector("agent", agent_collector)
    
    # Configure and register inter-agent metrics collector
    interagent_config = InterAgentMetricsConfig(
        sampling_interval_ms=1000,            # Sample every 1 second
        metrics_storage_path=os.path.join(output_dir, "interagent_metrics.json"),
        track_message_volume=True,
        track_coordination_overhead=True,
        # Queue-based processing configuration
        metrics_queue_size=2000,              # Medium-sized queue
        metrics_batch_size=20,                # Smaller batches for faster processing
        metrics_flush_interval_ms=200,        # Moderate flush interval
        sampling_rate=0.9                     # Sample 90% of interagent metrics
    )
    interagent_collector = InterAgentMetricsCollector(interagent_config)
    registry.register_collector("interagent", interagent_collector)
    
    print("Starting metrics collectors with dedicated processing threads...")
    # Start all collectors
    registry.start_all_collectors()
    
    return registry


# Simple LangGraph example nodes with metrics instrumentation
def create_instrumented_agent(agent_id: str, model_name: str = "gpt-3.5-turbo"):
    """Create an agent with metrics instrumentation."""
    
    # Get the metrics collectors
    registry = MetricsRegistry()
    agent_collector = registry.get_collector("agent")
    system_collector = registry.get_collector("system")
    
    # Create the LLM with a timeout
    llm = ChatOpenAI(
        model=model_name, 
        base_url=os.getenv("BASE_URL"), 
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=30,  # Add a 30-second timeout
        max_retries=2
    )
    
    # Define the agent function with metrics tracking
    def agent_node(state: Dict[str, Any]):
        """Agent node implementation with metrics instrumentation."""
        messages = state.get("messages", [])
        
        print(f"\n[DEBUG] {agent_id} processing message, queue length: {len(messages)}")
        
        # Record queue depth
        if system_collector:
            system_collector.record_queue_depth(f"agent_{agent_id}_queue", len(messages))
        
        print(f"[DEBUG] {agent_id} processing message, queue length: {len(messages)}")
        # Skip if no messages
        if not messages:
            return {"messages": messages}
        
        # Get the last message
        last_message = messages[-1]
        
        # Track the start time for latency calculation
        start_time = time.time()
        
        try:
            print(f"[DEBUG] {agent_id} calling LLM API with message: {last_message.content[:50]}...")
            
            # Run the LLM
            ai_message = llm.invoke([HumanMessage(content=last_message.content)])
            
            # Calculate metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            print(f"[DEBUG] {agent_id} received response in {latency_ms:.2f}ms")
            
            # Track token usage and latency
            if agent_collector:
                # Note: In a real implementation, you'd get actual token counts
                estimated_prompt_tokens = len(last_message.content) / 4  # Rough estimate
                estimated_completion_tokens = len(ai_message.content) / 4  # Rough estimate
                
                agent_collector.record_llm_usage(
                    agent_id=agent_id,
                    model_name=model_name,
                    prompt_tokens=estimated_prompt_tokens,
                    completion_tokens=estimated_completion_tokens,
                    latency_ms=latency_ms,
                    tags={"operation": "agent_response"}
                )
            
            # Track operation latency for system metrics
            if system_collector:
                system_collector.record_latency(
                    operation_name=f"agent_{agent_id}_response",
                    latency_ms=latency_ms,
                    tags={"agent_id": agent_id, "model": model_name}
                )
                
                # Record task completion
                system_collector.record_task_completion(
                    task_type="agent_response",
                    duration_ms=latency_ms,
                    tags={"agent_id": agent_id}
                )
            
            # Append the AI message to the messages list
            messages.append(AIMessage(content=ai_message.content))
            
            return {"messages": messages}
            
        except Exception as e:
            print(f"[ERROR] {agent_id} encountered an error: {str(e)}")
            
            # Record error
            if agent_collector:
                interagent_collector = registry.get_collector("interagent")
                if interagent_collector:
                    interagent_collector.record_error(
                        agent_id=agent_id,
                        error_type="llm_error",
                        severity="error",
                        tags={"error_message": str(e)}
                    )
            
            # Add a fallback response instead of getting stuck
            messages.append(AIMessage(content=f"I'm sorry, I encountered an error: {str(e)}. Let's try to continue anyway."))
            
            return {"messages": messages}
    
    return agent_node


def message_router(state: Dict[str, Any]) -> str:
    """Route messages to the appropriate agent."""
    # Get message count to alternate between agents
    message_count = len(state.get("messages", []))
    
    # Use simple round-robin routing
    if message_count % 2 == 0:
        return "agent_a"
    else:
        return "agent_b"


# LangGraph coordinator node with metrics instrumentation
def coordinator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinator node implementation with metrics instrumentation."""
    # Get the metrics collectors
    registry = MetricsRegistry()
    interagent_collector = registry.get_collector("interagent")
    
    messages = state.get("messages", [])
    
    print(f"[DEBUG] Coordinator processing {len(messages)} messages")
    
    # Skip if fewer than 2 messages
    if len(messages) < 2:
        return {"messages": messages, "coordination": []}
    
    # Get the latest messages
    agent_a_message = None
    agent_b_message = None
    
    for msg in reversed(messages):
        if hasattr(msg, "sender") and msg.sender == "agent_a" and agent_a_message is None:
            agent_a_message = msg
        elif hasattr(msg, "sender") and msg.sender == "agent_b" and agent_b_message is None:
            agent_b_message = msg
        
        if agent_a_message and agent_b_message:
            break
    
    # Track coordination event
    if interagent_collector and agent_a_message and agent_b_message:
        start_time = time.time()
        
        # Simulate coordination work
        time.sleep(0.1)  # Simulated coordination latency
        
        end_time = time.time()
        coordination_latency_ms = (end_time - start_time) * 1000
        
        interagent_collector.record_coordination_event(
            coordination_type="message_coordination",
            agents_involved=["agent_a", "agent_b"],
            latency_ms=coordination_latency_ms,
            outcome="success",
            tags={"message_count": len(messages)}
        )
        
        # Record message passing
        interagent_collector.record_message(
            source_agent_id="agent_a",
            target_agent_id="agent_b",
            message_type="coordination",
            message_size_bytes=len(str(agent_a_message.content)) if agent_a_message else 0,
            latency_ms=coordination_latency_ms / 2,  # Just an estimate
            tags={"coordination": "true"}
        )
    
    # Update state with coordination record
    coordination = state.get("coordination", [])
    coordination.append({
        "timestamp": datetime.now().isoformat(),
        "agent_a_message": agent_a_message.content if agent_a_message else None,
        "agent_b_message": agent_b_message.content if agent_b_message else None
    })
    
    return {"messages": messages, "coordination": coordination}


def build_instrumented_graph():
    """Build a LangGraph with metrics instrumentation."""
    # Define the state schema
    from typing import TypedDict, List as ListType
    
    class State(TypedDict):
        messages: ListType[Any]
        coordination: ListType[Dict[str, Any]]
    
    # Initialize the state graph
    workflow = StateGraph(State)
    
    # Add our nodes
    workflow.add_node("agent_a", create_instrumented_agent("agent_a"))
    workflow.add_node("agent_b", create_instrumented_agent("agent_b"))
    workflow.add_node("coordinator", coordinator_node)
    
    # Add the edges
    workflow.set_entry_point("agent_a")
    workflow.add_edge("agent_a", "coordinator")
    
    # Define end condition function
    def should_end(state):
        if len(state.get("messages", [])) >= 10:
            return "END"
        return message_router(state)
    
    # Use conditional edges for routing with end condition
    workflow.add_conditional_edges(
        "coordinator",
        should_end,
        {
            "agent_a": "agent_a",
            "agent_b": "agent_b",
            "END": END
        }
    )
    workflow.add_edge("agent_b", "coordinator")
    
    # Compile the graph
    instrumented_app = workflow.compile()
    
    return instrumented_app


async def run_example():
    """Run the example workflow and collect metrics."""
    # Set up the metrics framework
    registry = setup_metrics()
    
    print("Building instrumented LangGraph workflow...")
    app = build_instrumented_graph()
    
    print("Running workflow with metrics collection...")
    # Create initial state with a human message
    initial_state = {
        "messages": [HumanMessage(content="Tell me about multi-agent systems and their benefits.")]
    }
    
    try:
        # Run the workflow with a timeout
        print("Starting workflow execution...")
        
        # Create a task with timeout
        try:
            test = app.invoke(initial_state)
            print(test)
            result = await asyncio.wait_for(
                app.ainvoke(initial_state), 
                timeout=120  # 2-minute timeout
            )
            print("Workflow completed successfully!")
        except asyncio.TimeoutError:
            print("Workflow execution timed out after 120 seconds!")
            # Create a minimal result for analysis
            result = {
                "messages": initial_state["messages"] + [
                    AIMessage(content="Workflow execution timed out. Metrics collection will proceed anyway.")
                ],
                "coordination": []
            }
    except Exception as e:
        print(f"Error during workflow execution: {str(e)}")
        # Create a minimal result for analysis
        result = {
            "messages": initial_state["messages"] + [
                AIMessage(content=f"Workflow failed with error: {str(e)}")
            ],
            "coordination": []
        }
    
    # Export the metrics
    print("Exporting collected metrics...")
    registry.export_all("json", "metrics_output")
    
    # Stop metrics collection
    registry.stop_all_collectors()
    
    # Print final messages
    print("\nFinal Messages:")
    for i, message in enumerate(result["messages"]):
        role = "Human" if isinstance(message, HumanMessage) else "AI"
        print(f"{role} {i}: {message.content[:100]}...")
    
    # Analyze and visualize some metrics
    print("\nAnalyzing metrics...")
    analyze_metrics()


def analyze_metrics():
    """Analyze and visualize collected metrics."""
    # Get the metrics registry
    registry = MetricsRegistry()
    
    # Get the metrics collectors
    system_collector = registry.get_collector("system")
    agent_collector = registry.get_collector("agent")
    
    if not system_collector or not agent_collector:
        print("Metrics collectors not available")
        return
    
    # Get system latency metrics
    latency_metrics = system_collector.get_metrics(
        metric_names=["system.latency.agent_a_response", "system.latency.agent_b_response"]
    )
    
    # Get agent token usage metrics
    token_metrics = agent_collector.get_metrics()
    
    # Plot latency metrics
    if latency_metrics:
        try:
            plt.figure(figsize=(12, 6))
            
            for metric_name, data_points in latency_metrics.items():
                if not data_points:
                    continue
                    
                agent_id = metric_name.split('.')[-1].split('_')[1]
                
                timestamps = [datetime.fromisoformat(dp['timestamp']) for dp in data_points]
                values = [dp['value'] for dp in data_points]
                
                plt.plot(timestamps, values, label=f"Agent {agent_id}")
            
            plt.xlabel("Time")
            plt.ylabel("Latency (ms)")
            plt.title("Agent Response Latency")
            plt.legend()
            plt.tight_layout()
            plt.savefig("metrics_output/latency_analysis.png")
            print(f"Latency analysis chart saved to metrics_output/latency_analysis.png")
        except Exception as e:
            print(f"Error creating latency chart: {str(e)}")
    
    # Print summary statistics
    print("\nSystem Metrics Summary:")
    
    # Throughput
    throughput = system_collector.get_throughput("agent_response")
    print(f"Overall throughput: {throughput:.2f} tasks/second")
    
    # Latency percentiles for each agent
    for agent_id in ["agent_a", "agent_b"]:
        percentiles = system_collector.get_latency_percentiles(f"agent_{agent_id}_response")
        if percentiles:
            print(f"\n{agent_id.upper()} Latency Percentiles:")
            for p, value in percentiles.items():
                print(f"  p{p}: {value:.2f} ms")


def build_mock_test_graph():
    """Build a simplified test graph that doesn't use real LLMs for debugging."""
    # Define the state schema
    from typing import TypedDict, List as ListType
    
    class State(TypedDict):
        messages: ListType[Any]
        coordination: ListType[Dict[str, Any]]
    
    # Initialize the state graph
    workflow = StateGraph(State)
    
    # Define a simple mock agent
    def mock_agent(state: Dict[str, Any], agent_id: str):
        """Simple mock agent that doesn't use LLM API."""
        messages = state.get("messages", [])
        
        print(f"[DEBUG] {agent_id} mock processing")
        
        if not messages:
            return {"messages": messages}
            
        last_message = messages[-1]
        mock_response = f"This is a mock response from {agent_id}: Testing the metrics framework without real API calls."
        
        # Add mock response
        messages.append(AIMessage(content=mock_response))
        
        return {"messages": messages}
    
    # Add our nodes
    workflow.add_node("agent_a", lambda state: mock_agent(state, "agent_a"))
    workflow.add_node("agent_b", lambda state: mock_agent(state, "agent_b"))
    workflow.add_node("coordinator", coordinator_node)
    
    # Add the edges
    workflow.set_entry_point("agent_a")
    workflow.add_edge("agent_a", "coordinator")
    
    # Define end condition function
    def should_end(state):
        if len(state.get("messages", [])) >= 6:  # Shorter loop for testing
            return "END"
        return message_router(state)
    
    # Use conditional edges for routing with end condition
    workflow.add_conditional_edges(
        "coordinator",
        should_end,
        {
            "agent_a": "agent_a",
            "agent_b": "agent_b",
            "END": END
        }
    )
    workflow.add_edge("agent_b", "coordinator")
    
    # Compile the graph
    instrumented_app = workflow.compile()
    
    return instrumented_app


async def run_mock_test():
    """Run the mock test workflow for debugging."""
    # Set up the metrics framework
    registry = setup_metrics("mock_metrics_output")
    
    # Get system collector for direct testing
    system_collector = registry.get_collector("system")
    
    print("Building mock test workflow...")
    app = build_mock_test_graph()
    
    # Explicitly test queue depth recording
    print("Testing queue depth recording directly...")
    try:
        for i in range(5):
            print(f"Recording queue depth {i}...")
            system_collector.record_queue_depth("test_queue", i)
            # Small delay to ensure we don't overwhelm any locks
            await asyncio.sleep(0.1)
        print("Queue depth recording test completed successfully!")
    except Exception as e:
        print(f"Error during queue depth test: {str(e)}")
    
    print("Running mock workflow with metrics collection...")
    initial_state = {
        "messages": [HumanMessage(content="This is a test message.")]
    }
    
    try:
        print("Starting mock workflow execution...")
        result = await app.ainvoke(initial_state)
        print("Mock workflow completed successfully!")
    except Exception as e:
        print(f"Error during mock workflow execution: {str(e)}")
        result = {
            "messages": [HumanMessage(content="Test failed.")]
        }
    
    # Export the metrics
    print("Exporting mock test metrics...")
    registry.export_all("json", "mock_metrics_output")
    
    # Stop metrics collection
    registry.stop_all_collectors()
    
    # Print final messages
    print("\nFinal Messages from mock test:")
    for i, message in enumerate(result.get("messages", [])):
        role = "Human" if isinstance(message, HumanMessage) else "AI"
        print(f"{role} {i}: {message.content[:100]}...")
    
    return True


if __name__ == "__main__":
    import sys
    
    # Parse command-line arguments
    use_real_llm = "--real" in sys.argv
    
    if use_real_llm:
        print("========================================")
        print("Running full example with REAL LLM API calls...")
        print("This may take a while and could fail if API keys aren't properly configured.")
        print("========================================")
        asyncio.run(run_example())
    else:
        print("========================================")
        print("Running MOCK test version without real API calls...")
        print("(Use --real flag to run with actual LLM API calls)")
        print("========================================")
        asyncio.run(run_mock_test()) 