"""
Example demonstrating how to use the Agent and Inter-Agent Metrics Collectors.

This example shows how to instrument a multi-agent system with the metrics
framework to collect detailed performance data.
"""

import os
import time
import random
from datetime import datetime
import threading
from typing import Dict, List, Any
import json

# Import the metrics framework
from benchmark.src.metrics import (
    MetricsRegistry,
    AgentMetricsCollector,
    InterAgentMetricsCollector,
    SystemMetricsCollector
)

# Import config classes
from benchmark.src.metrics.agent_metrics import AgentMetricsConfig
from benchmark.src.metrics.inter_agent_metrics import InterAgentMetricsConfig
from benchmark.src.metrics.system_metrics import SystemMetricsConfig

# Simulated agent class for the example
class SimulatedAgent:
    def __init__(self, agent_id: str, metrics_registry: MetricsRegistry):
        self.agent_id = agent_id
        self.registry = metrics_registry
        self.agent_metrics = metrics_registry.get_collector("agent")
        self.interagent_metrics = metrics_registry.get_collector("interagent")
        self.system_metrics = metrics_registry.get_collector("system")
        self.memory_size = 0
        self.available_tools = ["search", "calculator", "knowledge_base", "executor"]
        
    def simulate_llm_call(self, prompt: str) -> str:
        """Simulate an LLM API call with instrumentation."""
        # Start timer for latency calculation
        start_time = time.time()
        
        # Simulate API call duration
        time.sleep(random.uniform(0.2, 1.5))
        
        # Simulate a response
        response = f"Simulated response from agent {self.agent_id}"
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.agent_metrics:
            # Estimate token counts (in a real system, you'd get this from the API)
            prompt_tokens = len(prompt) // 4
            completion_tokens = len(response) // 4
            
            # Record LLM usage metrics
            self.agent_metrics.record_llm_usage(
                agent_id=self.agent_id,
                model_name="gpt-3.5-turbo",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                tags={"operation": "query"}
            )
        
        # Record system metrics if available
        if self.system_metrics:
            self.system_metrics.record_latency(
                operation_name=f"agent_{self.agent_id}_llm",
                latency_ms=latency_ms,
                tags={"agent_id": self.agent_id}
            )
            
            # Record task completion
            self.system_metrics.record_task_completion(
                task_type="llm_call",
                duration_ms=latency_ms,
                tags={"agent_id": self.agent_id}
            )
        
        return response
    
    def use_tool(self, tool_name: str, args: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate a tool call with instrumentation."""
        # Start timer
        start_time = time.time()
        
        # Simulate tool execution
        time.sleep(random.uniform(0.1, 0.8))
        
        # Simulate success or failure
        success = random.random() > 0.2  # 80% success rate
        
        # Simulate result
        result = {"success": success, "data": f"Result from {tool_name}"} if success else {"success": False, "error": "Tool execution failed"}
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.agent_metrics:
            self.agent_metrics.record_tool_usage(
                agent_id=self.agent_id,
                tool_name=tool_name,
                success=success,
                latency_ms=latency_ms,
                args=args,
                tags={"execution_mode": "sync"}
            )
        
        return result
    
    def update_memory(self, content: str) -> None:
        """Simulate a memory update with instrumentation."""
        # Start timer
        start_time = time.time()
        
        # Simulate memory operation
        time.sleep(random.uniform(0.05, 0.2))
        
        # Update simulated memory size
        content_size_bytes = len(content.encode('utf-8'))
        self.memory_size += content_size_bytes
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.agent_metrics:
            # Record the memory operation
            self.agent_metrics.record_memory_operation(
                agent_id=self.agent_id,
                operation_type="write",
                latency_ms=latency_ms,
                operation_size_bytes=content_size_bytes,
                success=True,
                tags={"memory_type": "working_memory"}
            )
            
            # Record the updated memory size
            self.agent_metrics.record_memory_size(
                agent_id=self.agent_id,
                memory_type="working_memory",
                size_bytes=self.memory_size,
                tags={"persistent": "false"}
            )
    
    def make_decision(self, options: List[str]) -> str:
        """Simulate a decision-making process with instrumentation."""
        # Start timer
        start_time = time.time()
        
        # Simulate decision-making
        time.sleep(random.uniform(0.1, 0.3))
        
        # Select an option
        selected_option = random.choice(options)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.agent_metrics:
            self.agent_metrics.record_decision(
                agent_id=self.agent_id,
                decision_type="tool_selection",
                latency_ms=latency_ms,
                options_count=len(options),
                selected_option=selected_option,
                tags={"decision_strategy": "probabilistic"}
            )
        
        return selected_option
    
    def send_message(self, target_agent_id: str, message: str) -> bool:
        """Send a message to another agent with instrumentation."""
        # Start timer
        start_time = time.time()
        
        # Simulate message sending
        time.sleep(random.uniform(0.05, 0.3))
        
        # Simulate success or failure
        success = random.random() > 0.1  # 90% success rate
        
        # Calculate message size and latency
        message_size_bytes = len(message.encode('utf-8'))
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.interagent_metrics:
            self.interagent_metrics.record_message(
                source_agent_id=self.agent_id,
                target_agent_id=target_agent_id,
                message_type="request" if "?" in message else "inform",
                message_size_bytes=message_size_bytes,
                latency_ms=latency_ms,
                tags={"priority": "normal"}
            )
        
        return success
    
    def coordinate_task(self, other_agent_ids: List[str], task_description: str) -> bool:
        """Simulate a coordination event with instrumentation."""
        # Start timer
        start_time = time.time()
        
        # Simulate coordination
        time.sleep(random.uniform(0.2, 1.0))
        
        # Simulate success or failure
        success = random.random() > 0.15  # 85% success rate
        outcome = "success" if success else "failure"
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.interagent_metrics:
            self.interagent_metrics.record_coordination_event(
                coordination_type="task_assignment",
                agents_involved=[self.agent_id] + other_agent_ids,
                latency_ms=latency_ms,
                outcome=outcome,
                tags={"task_type": "data_processing"}
            )
            
            # Record decision consistency
            is_consistent = random.random() > 0.3  # 70% consistency rate
            self.interagent_metrics.record_decision_consistency(
                decision_id=f"task_{int(time.time())}",
                agents_involved=[self.agent_id] + other_agent_ids,
                consistent=is_consistent,
                decision_type="task_priority",
                tags={"context": "planning"}
            )
        
        return success
    
    def hand_off_task(self, target_agent_id: str, task_id: str, context: str) -> bool:
        """Simulate a task handoff with instrumentation."""
        # Start timer
        start_time = time.time()
        
        # Simulate handoff
        time.sleep(random.uniform(0.1, 0.5))
        
        # Simulate success or failure
        success = random.random() > 0.2  # 80% success rate
        
        # Calculate context size and latency
        context_size_bytes = len(context.encode('utf-8'))
        latency_ms = (time.time() - start_time) * 1000
        
        # Record metrics if available
        if self.interagent_metrics:
            self.interagent_metrics.record_task_handoff(
                task_id=task_id,
                source_agent_id=self.agent_id,
                target_agent_id=target_agent_id,
                handoff_latency_ms=latency_ms,
                context_size_bytes=context_size_bytes,
                success=success,
                tags={"handoff_type": "delegation"}
            )
        
        return success
    
    def simulate_error(self, error_type: str, severity: str, propagate_to: List[str] = None) -> None:
        """Simulate an error with instrumentation."""
        # Record metrics if available
        if self.interagent_metrics:
            # Simulate recovery time
            recovery_latency_ms = random.uniform(500, 3000) if random.random() > 0.3 else None
            
            self.interagent_metrics.record_error(
                agent_id=self.agent_id,
                error_type=error_type,
                severity=severity,
                related_agents=propagate_to,
                recovery_latency_ms=recovery_latency_ms,
                tags={"source": "simulation"}
            )

# Set up the metrics framework
def setup_metrics(output_dir: str = "metrics_output") -> MetricsRegistry:
    """Set up all metrics collectors for the example."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the metrics registry
    registry = MetricsRegistry()
    
    # Configure and register agent metrics collector
    agent_config = AgentMetricsConfig(
        sampling_interval_ms=1000,  # Sample every 1 second
        metrics_storage_path=os.path.join(output_dir, "agent.json"),  # Explicitly set file name
        track_token_usage=True,
        track_model_latency=True,
        track_memory_operations=True,
        track_tool_usage=True,
        track_tool_success_rate=True,
        track_memory_growth=True,
        # Queue-based processing configuration
        metrics_queue_size=10000,
        metrics_batch_size=100,
        metrics_flush_interval_ms=250,
        sampling_rate=1.0
    )
    agent_collector = AgentMetricsCollector(agent_config)
    registry.register_collector("agent", agent_collector)
    
    # Configure and register inter-agent metrics collector
    interagent_config = InterAgentMetricsConfig(
        sampling_interval_ms=1000,  # Sample every 1 second
        metrics_storage_path=os.path.join(output_dir, "interagent_metrics.json"),
        track_message_volume=True,
        track_message_size=True,
        track_coordination_overhead=True,
        track_decision_consistency=True,
        track_handoff_efficiency=True,
        track_error_rates=True,
        track_error_propagation=True,
        # Queue-based processing configuration
        metrics_queue_size=2000,
        metrics_batch_size=20,
        metrics_flush_interval_ms=200,
        sampling_rate=0.9
    )
    interagent_collector = InterAgentMetricsCollector(interagent_config)
    registry.register_collector("interagent", interagent_collector)
    
    # Configure and register system metrics collector
    system_config = SystemMetricsConfig(
        sampling_interval_ms=5000,  # Sample every 5 seconds
        metrics_storage_path=os.path.join(output_dir, "system_metrics.json"),
        monitor_cpu=True,
        monitor_memory=True,
        # Queue-based processing configuration
        metrics_queue_size=5000,
        metrics_batch_size=50,
        metrics_flush_interval_ms=100,
        sampling_rate=0.8
    )
    system_collector = SystemMetricsCollector(system_config)
    registry.register_collector("system", system_collector)
    
    print("Starting metrics collectors with dedicated processing threads...")
    # Start all collectors
    registry.start_all_collectors()
    
    return registry


def run_agent_simulation(duration_seconds: int = 30) -> None:
    """Run a simulation of multiple agents interacting with metrics collection."""
    print("Setting up metrics collection...")
    registry = setup_metrics()
    
    # Create simulated agents
    print("Creating simulated agents...")
    agents = [
        SimulatedAgent("research_agent", registry),
        SimulatedAgent("executor_agent", registry),
        SimulatedAgent("coordination_agent", registry)
    ]
    
    # Record start time
    start_time = time.time()
    end_time = start_time + duration_seconds
    
    print(f"Starting simulation (will run for {duration_seconds} seconds)...")
    
    # Simulation loop
    while time.time() < end_time:
        # Select a random agent to perform actions
        agent = random.choice(agents)
        
        # Randomly select an action to perform
        action = random.choice([
            "llm_call", "tool_use", "memory_update", "make_decision", 
            "send_message", "coordinate_task", "hand_off_task", "simulate_error"
        ])
        
        if action == "llm_call":
            prompt = f"This is a simulated prompt from {agent.agent_id} at {datetime.now()}"
            agent.simulate_llm_call(prompt)
            
        elif action == "tool_use":
            tool_name = random.choice(agent.available_tools)
            args = {"query": f"Test query from {agent.agent_id}", "options": {"detailed": True}}
            agent.use_tool(tool_name, args)
            
        elif action == "memory_update":
            content = f"Memory update from {agent.agent_id} at {datetime.now()}"
            agent.update_memory(content)
            
        elif action == "make_decision":
            options = agent.available_tools
            agent.make_decision(options)
            
        elif action == "send_message":
            # Select a random target agent that isn't this one
            target_agents = [a for a in agents if a.agent_id != agent.agent_id]
            if target_agents:
                target = random.choice(target_agents)
                message = f"Message from {agent.agent_id} to {target.agent_id} at {datetime.now()}"
                agent.send_message(target.agent_id, message)
                
        elif action == "coordinate_task":
            # Coordinate with other agents
            other_agent_ids = [a.agent_id for a in agents if a.agent_id != agent.agent_id]
            task_description = f"Coordination task initiated by {agent.agent_id} at {datetime.now()}"
            agent.coordinate_task(other_agent_ids, task_description)
            
        elif action == "hand_off_task":
            # Hand off task to another agent
            target_agents = [a for a in agents if a.agent_id != agent.agent_id]
            if target_agents:
                target = random.choice(target_agents)
                task_id = f"task_{int(time.time())}"
                context = f"Task context for {task_id} created by {agent.agent_id}"
                agent.hand_off_task(target.agent_id, task_id, context)
                
        elif action == "simulate_error":
            # Simulate an error (less frequently)
            if random.random() < 0.2:  # Only 20% chance to generate an error
                error_type = random.choice(["api_error", "timeout", "validation_error", "resource_error"])
                severity = random.choice(["low", "medium", "high", "critical"])
                
                # Randomly decide if error propagates to other agents
                propagate = random.random() < 0.3  # 30% chance to propagate
                propagate_to = []
                if propagate:
                    propagate_to = [a.agent_id for a in random.sample(agents, random.randint(1, len(agents)-1)) 
                                   if a.agent_id != agent.agent_id]
                    
                agent.simulate_error(error_type, severity, propagate_to)
        
        # Sleep briefly to prevent overwhelming the system
        time.sleep(random.uniform(0.05, 0.2))
    
    # Print summary statistics
    print("\nSimulation complete. Collecting final metrics...")
    
    # Get agent metrics
    agent_metrics = registry.get_collector("agent")
    if agent_metrics:
        print(f"Agent collector found: {agent_metrics.__class__.__name__}")
        
        # Directly add some metrics data points for testing
        test_agent = agents[0]
        for i in range(10):
            agent_metrics.collect_point(
                "agent.llm.tokens.total", 
                100 + i*10, 
                {"agent_id": test_agent.agent_id, "model_name": "gpt-3.5-turbo"}
            )
            
            agent_metrics.collect_point(
                "agent.tool.usage", 
                1, 
                {"agent_id": test_agent.agent_id, "tool_name": "search"}
            )
        
        # Allow time for metrics processing
        time.sleep(1)
        
        # Get metrics for each agent
        for agent_id in ["research_agent", "executor_agent", "coordination_agent"]:
            print(f"\n{agent_id} Metrics:")
            token_usage = agent_metrics.get_token_usage(agent_id)
            tool_usage = agent_metrics.get_tool_usage_stats(agent_id)
            
            print(f"  - Total tokens: {token_usage.get('total_tokens', 0) if token_usage else 0}")
            print(f"  - Total tool usages: {tool_usage.get('total_usages', 0) if tool_usage else 0}")
    else:
        print("Warning: Agent metrics collector not found!")
    
    # Get inter-agent metrics
    interagent_metrics = registry.get_collector("interagent")
    if interagent_metrics:
        message_volume = interagent_metrics.get_message_volume()
        coordination_stats = interagent_metrics.get_coordination_overhead()
        
        print("\nInter-Agent Metrics:")
        print(f"  - Total messages: {message_volume.get('total_messages', 0)}")
        print(f"  - Total coordination events: {coordination_stats.get('total_events', 0)}")
    
    # Get system metrics
    system_metrics = registry.get_collector("system")
    if system_metrics:
        throughput = system_metrics.get_throughput("llm_call")
        
        print("\nSystem Metrics:")
        print(f"  - LLM calls throughput: {throughput:.2f} calls/second")
    
    # Export all metrics before accessing other collectors
    print("\nExporting metrics to disk...")
    
    # First export agent metrics directly
    if agent_metrics:
        print("Exporting agent metrics directly...")
        
        # Use absolute path
        current_dir = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        metrics_output_dir = os.path.join(project_root, "metrics_output")
        agent_path = os.path.join(metrics_output_dir, "agent.json")
        
        print(f"Absolute path for agent metrics: {agent_path}")
        os.makedirs(metrics_output_dir, exist_ok=True)
        
        # Manually add some test metrics for demonstration
        for i in range(5):
            agent_metrics.collect_point(
                "agent.test.metric", 
                100 + i*10,
                {"agent_id": "test_agent", "test_tag": "value"}
            )
            
        # Allow time for processing
        time.sleep(0.5)
            
        # Export agent metrics by writing data directly
        try:
            metrics_data = {
                "token_usage": {},
                "tool_usage": {},
                "test_metrics": []
            }
            
            # Collect token usage for each agent
            for agent_id in ["research_agent", "executor_agent", "coordination_agent"]:
                token_data = agent_metrics.get_token_usage(agent_id)
                tool_data = agent_metrics.get_tool_usage_stats(agent_id)
                if token_data:
                    metrics_data["token_usage"][agent_id] = token_data
                if tool_data:
                    metrics_data["tool_usage"][agent_id] = tool_data
                    
            # Write to file
            with open(agent_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
            if os.path.exists(agent_path):
                print(f"Successfully exported agent metrics to {agent_path}")
            else:
                print(f"Export path {agent_path} not found after export!")
        except Exception as e:
            print(f"Error exporting agent metrics: {str(e)}")
            
    # Export all metrics via registry
    registry.export_all("json", "metrics_output")
    
    # Stop all collectors
    print("Stopping collectors...")
    registry.stop_all_collectors()
    
    print("Done!")


if __name__ == "__main__":
    # Run the simulation
    run_agent_simulation(duration_seconds=30) 