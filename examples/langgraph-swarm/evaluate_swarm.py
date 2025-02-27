# This evaluation script is adapted from the official MMLU repository:
# https://github.com/hendrycks/test
#
# Reference:
# @article{hendrycks2021measuring,
#     title={Measuring Massive Multitask Language Understanding},
#     author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},  # noqa: E501
#     journal={Proceedings of the International Conference on Learning Representations (ICLR)},
#     year={2021}
# }


from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from datasets import load_dataset
import numpy as np
import argparse


def format_example(question, choices, include_answer=False, answer=None):
    """Format a single example following MMLU format"""
    prompt = question + "\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65 + i)}. {choice}\n"
    prompt += "Answer:"
    if include_answer and answer is not None:
        prompt += f" {chr(65 + answer)}\n\n"
    return prompt


def gen_prompt(dev_set, subject, k=-1):
    """Generate few-shot prompt from dev set"""
    prompt = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    if k == -1:
        k = len(dev_set)
    for i in range(k):
        item = dev_set[i]
        prompt += format_example(item["question"], item["choices"], include_answer=True, answer=item["answer"])
    return prompt


def create_swarm_agent():
    """create your swarm agent"""
    # TODO: model except gpt-4 will cause error
    model = ChatOpenAI(model="gpt-4", temperature=0)

    # TODO: custom multi-agent debate
    math_expert = create_react_agent(
        model,
        [create_handoff_tool(agent_name="knowledge_expert")],
        prompt="You are a math expert. Always solve problems step by step.",
        name="math_expert",
    )

    knowledge_expert = create_react_agent(
        model,
        [create_handoff_tool(agent_name="math_expert")],
        prompt="You are a knowledge expert with broad understanding across various fields.",
        name="knowledge_expert",
    )

    checkpointer = InMemorySaver()
    workflow = create_swarm([math_expert, knowledge_expert], default_active_agent="knowledge_expert")
    return workflow.compile(checkpointer=checkpointer)


def format_conversation(result):
    """与 swarm_example.py 相同的格式化函数"""
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


def evaluate_subject(agent, subject, dataset, args):
    """evaluate a single subject"""
    correct = 0
    total = len(dataset["test"])
    config = {"configurable": {"thread_id": "1"}}

    for i, item in enumerate(dataset["test"]):
        # build few-shot prompt
        k = args.ntrain
        prompt_end = format_example(item["question"], item["choices"])
        train_prompt = gen_prompt(dataset["dev"], subject, k)
        prompt = train_prompt + prompt_end

        # get answer
        response = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config,
        )

        # TODO: save results to file
        # print("\n=== Agent Conversation ===")
        # print(format_conversation(response))
        # print("========================\n")

        for msg in reversed(response["messages"]):
            if msg.type == "ai" and msg.content:
                pred = msg.content.strip()[0]  # take the first character as the answer
                break

        correct_answer = chr(65 + item["answer"])
        is_correct = pred == correct_answer
        correct += is_correct

        print(f"Question {i + 1}/{total}: {'✓' if is_correct else '✗'} (Pred: {pred}, True: {correct_answer})")

    accuracy = correct / total
    print(f"\nAccuracy for {subject}: {accuracy:.3f}")
    return accuracy


def main(args):
    agent = create_swarm_agent()

    dataset = load_dataset("cais/mmlu", "elementary_mathematics")

    subject_data = {"test": dataset["test"], "dev": dataset["dev"], "validation": dataset["validation"]}

    acc = evaluate_subject(agent, "elementary_mathematics", subject_data, args)
    print(f"\nFinal accuracy: {acc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    args = parser.parse_args()
    main(args)
