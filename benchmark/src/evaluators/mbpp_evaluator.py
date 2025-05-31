"""
MBPP Evaluator
"""

from __future__ import annotations

import logging
import re
import time
import traceback
import uuid
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Tuple

from langsmith.evaluation import RunEvaluator
from langsmith.schemas import Run

from benchmark.src.evaluators.utils.sanitize import sanitize


class TimeoutError(Exception):
    """Raised when the sandboxed execution exceeds the time-limit."""


def run_with_timeout(func, args: tuple[Any, ...] = (), timeout: int = 15):
    """
    Execute `func(*args)` in a daemon thread.
    Raise `TimeoutError` if it runs longer than `timeout` seconds.
    """
    result: list[Any] = []

    def target():
        result.append(func(*args))

    thread = Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise TimeoutError(f"Execution timed out after {timeout}s")

    return result[0] if result else None


class MBPPEvaluator:
    """Evaluator for MBPP code-generation tasks."""

    def __init__(self, name: str = "mbpp", config: Dict[str, Any] | None = None):
        self.name = name
        self.config = config or {}

        # Default paths can be overridden via `config`
        self.data_path = self.config.get("data_path", f"benchmark/data/{name}.jsonl")
        self.log_path = self.config.get("log_path", f"benchmark/data/results/{name.upper()}")

        Path(self.log_path).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=f"{self.log_path}/evaluator.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        self.run_evaluator = RunEvaluator()


    def extract_code(self, text: str) -> str:
        """
        Best-effort extraction of Python code from agent output.

        Priority:
        1. Code under a "## Validated Code" heading.
        2. First ```python fenced block.
        3. Entire text (fallback) – `sanitize` will later trim Markdown.
        """
        validated = re.search(r"##\s*Validated Code\s*```python\s*([\s\S]*?)```", text, re.I)
        if validated:
            return validated.group(1).strip()

        fenced = re.search(r"```python\s*([\s\S]*?)```", text, re.I)
        if fenced:
            return fenced.group(1).strip()

        return text.strip()


    def check_solution(
        self,
        code: str,
        test: str,
        entry_point: str,
        test_imports: List[str] | None = None,
    ) -> Tuple[bool, str]:
        """
        Compile user code, run official MBPP `check()`.

        Returns:
            (passed: bool, message: str)
            `passed` is True iff all assertions succeed within the time-limit.
        """
        try:
            # Remove Markdown, ensure the target function exists
            code_clean = sanitize(code=code, entrypoint=entry_point)

            # Isolated global namespace
            env: Dict[str, Any] = {}
            exec(code_clean, env)

            # Execute additional import statements if provided
            for stmt in test_imports or []:
                exec(stmt, env)

            if entry_point not in env:
                raise ValueError(f"Function `{entry_point}` is missing in submitted code.")

            # Inject and run the official unit tests
            exec(test, env)
            check_fn = env["check"]

            run_with_timeout(check_fn, timeout=15)  # `check()` takes no args
            return True, "All tests passed"

        except TimeoutError as te:
            return False, str(te)
        except AssertionError as ae:
            return False, f"Assertion failed: {ae}"
        except Exception as exc:  # noqa: BLE001
            if self.config.get("verbose"):
                self.logger.error(traceback.format_exc())
            return False, f"Execution error: {exc}"

    def evaluate(self, problem: Dict[str, Any], run_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entrypoint called by benchmark_runner.

        Args:
            problem: A dict with MBPP fields (`prompt`, `test`, ...)
            run_result: Output from the MetaGPT agent chain (contains "final_answer").

        Returns:
            Dict with score, extracted answer, LangSmith run evaluation, etc.
        """
        final_answer_text = run_result.get("final_answer", "")
        extracted_code = self.extract_code(final_answer_text)

        passed, msg = self.check_solution(
            extracted_code,
            problem["test"],
            problem["entry_point"],
            problem.get("test_imports") or [],
        )
        score = 1.0 if passed else 0.0

        # Generate valid UUIDs for LangSmith
        run = Run(
            id=str(uuid.uuid4()),
            name=f"{self.name.upper()}_Evaluation",
            inputs={"problem": problem["problem"], "task_id": problem["id"]},
            outputs={
                "prediction": final_answer_text,
                "extracted_code": extracted_code,
                "expected": problem["solution"],
                "score": score,
                "message": msg,
                "passed": passed,
            },
            run_type="evaluation",
            start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
            trace_id=str(uuid.uuid4()),
        )
        run_eval = self.run_evaluator.evaluate_run(run=run)

        return {
            "final_answer": final_answer_text,
            "extracted_answer": extracted_code,
            "score": score,
            "message": msg,
            "run_evaluation": run_eval,
        }
