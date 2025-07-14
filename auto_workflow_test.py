# import asyncio
# import os
# import sys
# import traceback
# import json
#
# from dotenv import load_dotenv
#
# load_dotenv()
#
# def get_llm_config():
#     key = os.getenv("OPENAI_API_KEY")
#     base_url = os.getenv("OPENAI_API_BASE")
#     if not key or not base_url:
#         raise ValueError("OPENAI_API_KEY or OPENAI_API_BASE not set in environment variables.")
#     llm_config = OpenAILLMConfig(
#         model="gpt-4o-mini",
#         openai_key=key,
#         base_url=base_url
#     )
#     return llm_config
#
#
#
# def get_evaluator():
#     from mas_arena.evaluators.humaneval_evaluator import HumanEvalEvaluator
#     return HumanEvalEvaluator("humaneval", {})
#
#
# def get_humaneval_problem(idx=0, jsonl_path="data/humaneval_test.jsonl"):
#     # 获取当前脚本所在目录（假设脚本在项目根或 test/ 下）
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     # 拼接到项目根目录
#     project_root = os.path.abspath(os.path.join(base_dir, "."))
#     jsonl_full_path = os.path.join(project_root, jsonl_path)
#     with open(jsonl_full_path, "r", encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             if i == idx:
#                 item = json.loads(line)
#                 return item["prompt"], item["entry_point"]
#     raise IndexError(f"Index {idx} out of range for {jsonl_full_path}")
#
# async def test_workflow(workflow_cls, problem, entry_point):
#     llm_config = get_llm_config()
#     evaluator = get_evaluator()
#     workflow = workflow_cls("test_workflow", llm_config, evaluator)
#     try:
#         result = await workflow(problem, entry_point)
#         print("Workflow result:", result)
#     except Exception as e:
#         print("Error during workflow execution:")
#         traceback.print_exc()
#
# if __name__ == "__main__":
#     problem_idx = 3
#     from example.aflow.humaneval.optimization.round_3.graph import Workflow
#     #from example.aflow.humaneval.optimization.round_5.graph import Workflow
#
#     problem, entry_point = get_humaneval_problem(problem_idx)
#     print(f"Testing problem idx={problem_idx}, entry_point={entry_point}\nPrompt:\n{problem}")
#     asyncio.run(test_workflow(Workflow, problem, entry_point))