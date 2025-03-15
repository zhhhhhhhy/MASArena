# Multi-Agent Workflow with Adaptive Routing

This project implements a flexible multi-agent system using LangGraph that supports dynamic routing between agents based on configurable connection graphs and edge weights.

**BaseWorkflow**
A base workflow using LangGraph to create a fully connected network for multiple agents. This provides the foundation for agent communication and routing.

**AdjustableWorkflow**
Extends the BaseWorkflow with runtime adjustment capabilities, allowing for dynamic modification of agent connections (edges) and routing behavior.

- [x] **Connection-based Routing**: Given another graph that defines agent connections, the adjustable workflow will use that graph for routing.
- [x] **Weight-based Routing**: Given another graph with edge weights, the adjustable workflow adapts the edge weights to the new connections.

## How to run


```bash
# uv env setup
uv init

# install dependencies
uv add -r requirements.txt

# activate env
source .venv/bin/activate 

# env setup
cp .env.example .env # edit .env

# run tests
python -m src.test_adjustable_workflow_rigorous
python -m src.test_edge_weight_adaptation
... # more tests
```

## How Adaptive Routing Works

### 1. Connection Graph

The system uses a `ConnectionManager` class that maintains a directed graph representing the allowed connections between agents. Each agent can only route to agents that are connected to it in this graph.

```python
# Example: Creating a connection graph
connection_manager = ConnectionManager(agent_names=["A", "B", "C"])
connection_manager.set_connections("A", ["B", "C"])
connection_manager.set_connections("B", ["C"])
connection_manager.set_connections("C", [END])
```

### 2. Command-based Routing

The key to adaptive routing is the use of LangGraph's `Command` objects. Each agent node returns a Command that specifies:

1. Which agent to route to next (`goto`)
2. How to update the state (`update`)

```python
# Inside agent_node function
return Command(
    goto=next_agent,  # Dynamically determined based on connection graph
    update={"messages": state["messages"] + [ai_message]}
)
```

### 3. Dynamic Routing Logic

When an agent is called, it:

1. Examines the current state (messages, context)
2. Consults the connection graph to determine allowed next agents
3. Makes a decision about which agent to route to next
4. Returns a Command with the routing decision

```python
# Simplified agent node logic
def agent_node(state):
    # Get allowed next agents from connection graph
    allowed_next_agents = connection_manager.get_allowed_next_agents(agent_name)
    
    # Make routing decision based on message content and allowed connections
    next_agent = determine_next_agent(state, allowed_next_agents)
    
    # Return command with routing decision
    return Command(goto=next_agent, update={...})
```

### 4. Edge Weights for Probabilistic Routing

The system supports edge weights that influence routing decisions:

```python
# Setting connections with weights
workflow.set_connections("Dispatcher", ["ServiceA", "ServiceB"], weights=[0.7, 0.3])
```

These weights can be used to implement probabilistic routing, where agents are more likely to route to connections with higher weights.

### 5. Adapting from External Graphs

The workflow can adapt its routing behavior from external graphs:

```python
# Create an external graph with custom routing
external_graph = nx.DiGraph()
external_graph.add_edge("A", "B", weight=0.8)
external_graph.add_edge("A", "C", weight=0.2)

# Update the workflow with the external graph
workflow.update_weights_from_graph(external_graph)
```

## Implementation Details

### Command-based State Machine

The core of the adaptive routing is implemented as a state machine where:

- **States**: Agent nodes that process messages
- **Transitions**: Determined by the connection graph
- **Commands**: Control flow between states

Each agent returns a Command object that tells LangGraph where to route next:

```python
Command(
    goto=next_agent,  # Dynamic routing based on connection graph
    update={"messages": state["messages"] + [ai_message]}
)
```

### Runtime Adaptation

The workflow can be modified at runtime:

```python
# Add a new connection
workflow.add_connection("A", "D", weight=0.5)

# Remove a connection
workflow.remove_connection("A", "B")

# Update edge weights
workflow.set_edge_weight("A", "C", 0.8)
```

This allows for dynamic adaptation of the agent network based on changing requirements or external inputs.

## Visualization

The system includes visualization tools to inspect the connection graph:

```python
# Visualize the current connection graph
workflow.visualize_connections(title="Agent Connection Graph", output_file="graph.png")
```

This generates a visual representation of the agent connections and their weights.

## Usage Example

```python
from src.adjustable_workflow import AdjustableWorkflow
from langchain_core.messages import HumanMessage

# Create a workflow with specific agents
agent_names = ["Coordinator", "Researcher", "Writer", "Editor"]
workflow = AdjustableWorkflow(agent_names)

# Configure the connections
workflow.set_connections("Coordinator", ["Researcher", "Writer"], weights=[0.7, 0.3])
workflow.set_connections("Researcher", ["Writer"])
workflow.set_connections("Writer", ["Editor", END], weights=[0.9, 0.1])
workflow.set_connections("Editor", [END])

# Run the workflow
result = workflow.invoke({
    "messages": [HumanMessage(content="Write a report on renewable energy.")]
})
```


