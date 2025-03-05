# 目标：使用 LangGraph, langgraph-swarm-py, LangSmith, Open-Evals 来实现 MMLU 评测，并追踪智能体执行流程

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm
from datasets import load_dataset
from langsmith.evaluation import EvaluationResult, RunEvaluator
from langsmith import traceable
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
import os
import argparse
from dotenv import load_dotenv

load_dotenv()


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


@traceable
def create_swarm_agent():
    """Create a LangGraph swarm agent with LangSmith tracking"""
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("CUSTOM_API_KEY"),
        openai_api_base=os.getenv("CUSTOM_API_BASE"),
    )

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

    # compile a swarm to remember the previous interaction and the last active agent
    return workflow.compile(checkpointer=checkpointer)


def evaluate_subject(agent, subject, dataset, args):
    """Use LangSmith & Open-Evals to evaluate the model"""
    langsmith_client = RunEvaluator()

    # 创建 OpenEvals 评估器函数
    openevals_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",  # 添加 feedback_key
        model="gpt-4o-mini",
    )

    correct = 0
    total = len(dataset["test"])
    config = {"configurable": {"thread_id": "1"}}
    predictions = []
    ground_truths = []
    run_metadata = []

    for i, item in enumerate(dataset["test"]):
        k = args.ntrain
        prompt_end = format_example(item["question"], item["choices"])
        train_prompt = gen_prompt(dataset["dev"], subject, k)
        prompt = train_prompt + prompt_end

        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config)

        for msg in reversed(response["messages"]):
            if msg.type == "ai" and msg.content:
                pred = msg.content.strip()[0]  # take the first character as the answer
                break

        correct_answer = chr(65 + item["answer"])
        is_correct = pred == correct_answer
        correct += is_correct

        predictions.append(pred)
        ground_truths.append(correct_answer)

        # OpenEvals evaluation - 直接调用函数
        eval_result = openevals_evaluator(
            inputs={"question": item["question"]},
            outputs={"model_response": pred},
            reference_outputs={"correct_answer": correct_answer},
        )

        run_metadata.append(
            {
                "question": item["question"],
                "choices": item["choices"],
                "predicted": pred,
                "actual": correct_answer,
                "correct": is_correct,
                "subject": subject,
                "openeval_score": eval_result["score"],
                "openeval_feedback": eval_result.get("feedback", ""),
                "evaluation_type": "multiple_choice",
            }
        )

        print(f"Question {i + 1}/{total}: {'✓' if is_correct else '✗'} (Pred: {pred}, True: {correct_answer})")
        print(f"Open-Evals Score: {eval_result['score']}")

    langsmith_results = langsmith_client.evaluate_run(
        dataset_name=f"mmlu_eval_{subject}", predictions=predictions, ground_truths=ground_truths, metadata=run_metadata
    )
    print(f"\nLangSmith Eval Results: {langsmith_results}")

    accuracy = correct / total
    print(f"\nFinal Accuracy for {subject}: {accuracy:.3f}")
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
