"""
Single Agent System

This module implements a simple single-agent system that uses a single LLM
to solve problems directly.
"""

import nltk
nltk.download('punkt_tab')
import time
import uuid
import os
import re
import json
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from benchmark.src.agents.base import AgentSystem, AgentSystemRegistry

# Load environment variables
load_dotenv()


class SingleAgent(AgentSystem):
    """
    Single Agent System

    This agent system uses a single LLM to solve problems directly.
    """

    def __init__(self, name: str = "single_agent", config: Dict[str, Any] = None):
        """Initialize the Single Agent System"""
        super().__init__(name, config)
        self.config = config or {}
        self.evaluator_name = self.config.get("evaluator", "bbh")  # Default to bbh
        self.model_name = self.config.get("model_name") or os.getenv("MODEL_NAME", "qwen-plus")  # Use qwen-plus
        self.system_prompt = (
            self.config.get("system_prompt")
            or "You are an intelligent AI assistant specialized in solving complex problems step by step."
        )

        # Initialize evaluator and metrics collector through base class methods
        self._initialize_evaluator()
        self._initialize_metrics_collector()
        self.llm = ChatOpenAI(
            model_name=os.getenv("MODEL_NAME", "gpt-4.1-mini"),
        )

    # _create_prompt for BBH problems
    def _create_prompt(self, problem_type:str, problem_text: str, task_type: str) -> str:
        if problem_type == "math":
            return f"""
    Please solve the following problem carefully and step by step:

    {problem_text}

    For mathematical problems, make sure to:
    1. Break down the problem into simpler parts
    2. Solve each part methodically
    3. Check your work and verify your answer
    4. Provide your final answer in a clear format
    """
        # Agent system: single_agent
        # Accuracy: 80.00% (40/50)
        # Total duration: 176691ms
        elif problem_type == "bbh":
            return f"""
        You are a highly capable AI assistant tackling a problem from the **Big-Bench Hard (BBH)** benchmark.

        Task Type: {task_type}

        Your reply **MUST** contain only two HTML-like blocks and nothing else:

        <think>
        …your step-by-step reasoning goes ONLY here…
        </think>
        <answer>
        …final answer ONLY, in the exact format required below…
        </answer>

        Rules are as follows:

        1. Put **ALL** reasoning strictly inside the `<think> … </think>` block.
        2. Put the **FINAL** answer on **one single line** inside `<answer> … </answer>`,
        with no leading or trailing spaces.
        3. Output **absolutely nothing** outside those two blocks.
        4. Match the exact answer format for the task type you are solving.
        The BBH task families and their required answer literals are:

        • **boolean_expressions** 
            → `True` / `False`  
            Example ➜ `<answer>True</answer>`

        • **causal_judgement**, 
            **web_of_lies**, 
            **sports_understanding**,
            **navigate** 
            → `Yes` / `No`  
            Example ➜ `<answer>No</answer>`

        • **date_understanding**, 
            **disambiguation_qa**, 
            **hyperbaton**,  
            **temporal_sequences**, 
            **salient_translation_error_detection**,  
            **ruin_names**, 
            **snarks**, 
            **movie_recommendation**,  
            **geometric_shapes**,
            **penguins_in_a_table**, 
            **reasoning_about_colored_objects**,
            **logical_deduction_three_objects / five_objects / seven_objects**,  
            **tracking_shuffled_objects_three_objects / five_objects / seven_objects**   
            → exactly one option letter shown in the prompt, wrapped in parentheses like `(A)` … `(E)` (or any other letter supplied).  
            Example ➜ `<answer>(C)</answer>`

        • **formal_fallacies** → `valid` / `invalid`  
            Example ➜ `<answer>invalid</answer>`

        • **dyck_languages** → ONE bracket that completes the sequence:  
            `)`, `]`, `}}`, or `>`  
            Example ➜ `<answer>)</answer>`(if input is `< < [ ( ) ] >`,then you just need to output `>` to complete the sequence, rather than the whole sequence)

        • **word_sorting** → the words in strict alphabetical order, lower-case,  
            single-space separated  
            Example ➜ `<answer>apple boy zoo</answer>`

        • **multistep_arithmetic_two**, 
            **object_counting** 
            → an integer (may be negative) with no commas or spaces  
            Example ➜ `<answer>-330</answer>`

        • *If the problem statement presents its own answer schema, follow that
            schema **verbatim**—it overrides every rule above.*

        Problem:
        {problem_text}

        Remember: respond with **ONLY** the two blocks shown above—nothing else.
        """
        elif problem_type == "drop":
            return f"""
You are an expert reading-comprehension assistant tackling a question from the **DROP** benchmark
(Discrete Reasoning Over Paragraphs).

Your reply **MUST** contain **exactly two** XML-like blocks and nothing else:

<think>
…Write your step-by-step reasoning here…
</think>
<answer>
…ONLY the final answer here (no extra words, no units unless they are part of the answer)…
</answer>

Passage & Question:
{problem_text}

Remember:
1. Put **all** reasoning strictly inside <think> … </think>.
2. The <answer> block must contain only the short answer string required by the question,
   trimmed of leading/trailing spaces.
3. Output absolutely nothing outside those two blocks.
"""

        elif problem_type == "ifeval":
        # IFEval —— Instruction Following Evaluation
            return f"""
You are taking part in the **Instruction Following Evaluation (IFEval)** benchmark.

Read the user instruction carefully and produce ONE final response that satisfies *every*
constraint implied by the instruction IDs and by the instruction text itself.

### Rules
1. Think step-by-step **silently** – do **NOT** reveal your reasoning.
2. Follow all punctuation, formatting, length, highlighting and stylistic constraints exactly.
3. If an instruction forbids an element (e.g. no commas), *never* include it.
4. If an instruction sets a minimum (e.g. ≥ 300 words, ≥ 3 highlighted sections), be sure to exceed it.
5. Return **only** the finished response text – no explanations, no markdown fences, no extra whitespace.

---  USER INSTRUCTION  ---
{problem_text}
--------------------------------

Begin now.  Remember: output only the compliant answer.
"""


    def run_agent(self, problem: Dict[str, Any], problem_type: str, **kwargs) -> Dict[str, Any]:
        """
        Run the agent system on a given problem.

        This method implements the actual agent logic without handling evaluation or metrics.

        Args:
            problem: Dictionary containing the problem data
            problem_type: Type of problem (e.g., 'bbh')

        Returns:
            Dictionary of run results including messages with usage metadata
        """
        problem_text = problem["problem"]
        problem_id = str(problem.get("id", f"problem_{hash(problem_text)}"))
        task_type = re.sub(r'_\d+$', '', problem_id) if problem_id else ""
    
        # Initialize the language model
        llm = self.llm

        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._create_prompt(problem_type,problem_text, task_type)},
        ]

        # Get solution from LLM and track usage
        response = llm.invoke(messages)

        # Clean response content
        response_content = response.content.replace("\r\n", "\n").replace("\r", "\n").strip()
        try:
            response_content = response_content.encode("utf-8").decode("utf-8-sig")  # Remove BOM
        except UnicodeDecodeError:
            pass  # Ignore if already clean

        # print("模型返回结果:", response)  # Keep original print
        # print(f"[Debug] Cleaned response content: {repr(response_content)}")  # Debugging

        ai_message = response
        ai_message.name = "single_agent"

        # Return the response and message with usage metadata for the evaluate method
        return {
            "messages": [ai_message],
            "final_answer": response_content,  # Use cleaned content
        }


# Register the agent system
AgentSystemRegistry.register("single_agent", SingleAgent)
