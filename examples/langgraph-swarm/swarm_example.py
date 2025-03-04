from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm


def format_conversation(result):
    conversation = []
    for message in result["messages"]:
        if message.type == "human":
            conversation.append(f"User: {message.content}")
        elif message.type == "ai":
            if message.content:
                conversation.append(f"{message.name}: {message.content}")
        elif message.type == "tool":
            conversation.append(f"System: {message.content}")

    return "\n".join(conversation)


def main():
    model = ChatOpenAI(model="gpt-4")

    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    alice = create_react_agent(
        model,
        [add, create_handoff_tool(agent_name="Bob")],
        prompt="You are Alice, an addition expert.",
        name="Alice",
    )

    bob = create_react_agent(
        model,
        [create_handoff_tool(agent_name="Alice", description="Transfer to Alice, she can help with math")],
        prompt="You are Bob, you speak like a pirate.",
        name="Bob",
    )

    checkpointer = InMemorySaver()
    workflow = create_swarm([alice, bob], default_active_agent="Alice")
    app = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "1"}}

    print("\n=== Conversation 1 ===")
    turn_1 = app.invoke(
        {"messages": [{"role": "user", "content": "i'd like to speak to Bob"}]},
        config,
    )
    print(format_conversation(turn_1))

    print("\n=== Conversation 2 ===")
    turn_2 = app.invoke(
        {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
        config,
    )
    print(format_conversation(turn_2))


if __name__ == "__main__":
    main()
