import argparse
import os
from dotenv import load_dotenv

from mas_arena.evaluators.humaneval_evaluator import HumanEvalEvaluator
from mas_arena.optimizers.aflow.aflow_optimizer import AFlowOptimizer
from mas_arena.optimizers.aflow.aflow_experimental_config import EXPERIMENTAL_CONFIG

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")
OPTIMIZER_MODEL_NAME = os.getenv("OPTIMIZER_MODEL_NAME") or "gpt-4o"
EXECUTOR_MODEL_NAME = os.getenv("EXECUTOR_MODEL_NAME") or os.getenv("MODEL_NAME") or "gpt-4o-mini"


def main():
    parser = argparse.ArgumentParser(description="Run AFlow optimization")
    from mas_arena.evaluators import BENCHMARKS
    parser.add_argument(
        "--benchmark",
        type=str,
        default="humaneval",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run (default: math)",
    )

    parser.add_argument(
        "--graph_path",
        type=str,
        default="mas_arena/configs/aflow",
        help="Path to the AFlow graph configuration",
    )

    parser.add_argument(
        "--optimized_path",
        type=str,
        default="example/aflow/humaneval/optimization",
        help="Path to save the optimized AFlow graph",
    )

    parser.add_argument(
        "--validation_rounds",
        type=int,
        default=1,
        help="Number of validation rounds",
    )

    parser.add_argument(
        "--eval_rounds",
        type=int,
        default=1,
        help="Number of evaluation rounds",
    )

    parser.add_argument(
        "--max_rounds",
        type=int,
        default=3,
        help="Maximum number of optimization rounds",
    )
    # Parse arguments
    args = parser.parse_args()

    from mas_arena.agents import AgentSystemRegistry
    optimizer_config = {"model_name": OPTIMIZER_MODEL_NAME, "API_KEY": API_KEY,
                        "API_BASE": API_BASE}
    optimizer_agent = AgentSystemRegistry.get("single_agent", optimizer_config)
    executor_config = {"model_name": EXECUTOR_MODEL_NAME, "API_KEY": API_KEY, "API_BASE": API_BASE}
    executor_agent = AgentSystemRegistry.get("single_agent", executor_config)

    # load evaluator
    evaluator = HumanEvalEvaluator(args.benchmark, {})

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path=args.graph_path,
        optimized_path=args.optimized_path,
        optimizer_agent=optimizer_agent,
        executor_agent=executor_agent,
        validation_rounds=1,
        eval_rounds=1,
        max_rounds=3,
        **EXPERIMENTAL_CONFIG[args.benchmark]
    )

    optimizer.setup()
    # run optimization
    optimizer.optimize(evaluator)

    # run test
    optimizer.test(evaluator)


if __name__ == "__main__":
    main()
