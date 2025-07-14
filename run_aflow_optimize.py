import argparse
import os
from dotenv import load_dotenv

from mas_arena.evaluators.humaneval_evaluator import HumanEvalEvaluator
from mas_arena.optimizers.aflow_optimizer import AFlowOptimizer
from mas_arena.optimizers.aflow.aflow_experimental_config import EXPERIMENTAL_CONFIG

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_API_BASE = os.getenv("ANTHROPIC_API_BASE")


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
        "graph_path"
        , type=str,
        default="mas_arena/configs/aflow",
        help="Path to the AFlow graph configuration",
    )

    parser.add_argument(
        "optimized_path",
        type=str,
        default="example/aflow/humaneval/optimization2",
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
    optimizer_config = {"model_name": "claude-3-5-sonnet-latest", "API_KEY": ANTHROPIC_API_KEY,
                        "API_BASE": ANTHROPIC_API_BASE}
    optimizer_agent = AgentSystemRegistry.get("single_agent", optimizer_config)
    executor_config = {"model_name": "gpt-4o-mini", "API_KEY": OPENAI_API_KEY, "API_BASE": OPENAI_API_BASE}
    executor_agent = AgentSystemRegistry.get("single_agent", executor_config)

    # load evaluator
    evaluator = HumanEvalEvaluator(args.benchmark, {})

    # create optimizer
    optimizer = AFlowOptimizer(
        graph_path="mas_arena/configs/aflow",
        optimized_path="example/aflow/humaneval/optimization2",
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
