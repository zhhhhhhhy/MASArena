[x] **BaseWorkflow**: a base workflow using LangGraph to create a fully connected network for multiple agents (the same agent without tools or other complex components)
[] **AdjustableWorkflow**: Extends the base workflow with runtime adjustment capabilities, allowing for dynamic modification of agent connections (edges).
 1. Given another graph, which defined the agent connections, the adjustable workflow will use that graph for routing.
 2. Given another graph with edge weights, the adjustable workflow adapts the edge weights to the new connections.
[] **Example**: A demonstration of how to use the adjustable workflow with multiple agents.


