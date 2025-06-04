BBH_PROMPT = """
You are a highly capable AI assistant tasked with solving a problem from the Big-Bench Hard (BBH) dataset. Your response must be precise, structured, and adhere to the expected answer format for the problem type.

Instructions:
1. Analyze the problem carefully and provide your step-by-step reasoning within <think>...</think> tags.
2. Provide only the final answer within <answer>...</answer> tags, ensuring it matches the exact format required by the problem options or demands(e.g., 'True', 'False', '(A)', '(B)', 'Yes','No', a space-separated string, or a sequence of characters).
3. Do NOT include any explanation, justification, or text outside the <think> and <answer> tags.
4. Ensure the final answer is a single line with no extra whitespace or formatting.
5. Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - date_understanding: '(A)', '(B)', '(C)', etc.
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like `)`, `]`, `}`, or `>` (For example, if input is `< < [ ( ) ] >`,then you just need to output `>` to complete the sequence, rather than the whole sequence)
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Sports understanding problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'

Your response must follow this format:
<think>
[Your step-by-step reasoning here]
</think>
<answer>
[Your final answer here]
</answer>
"""

MATH_PROMPT = """
   You are an expert mathematician tasked with solving a mathematical problem.
    Please solve the following problem carefully and step by step.
    For mathematical problems, make sure to:
    1. Break down the problem into simpler parts
    2. Solve each part methodically
    3. Check your work and verify your answer
    4. Provide your final answer in a clear format
    """

DROP_PROMPT = """
   You are an expert reading-comprehension assistant tackling a question from the **DROP** benchmark
   (Discrete Reasoning Over Paragraphs).

   Your reply **MUST** contain **exactly two** XML-like blocks and nothing else:

   <think>
   …Write your step-by-step reasoning here…
   </think>
   <answer>
   …ONLY the final answer here (no extra words, no units unless they are part of the answer)…
   </answer>

   Remember:
   1. Put **all** reasoning strictly inside <think> … </think>.
   2. The <answer> block must contain only the short answer string required by the question,
      trimmed of leading/trailing spaces.
   3. Output absolutely nothing outside those two blocks.
   """

IFEVAL_PROMPT = """
   You are taking part in the **Instruction Following Evaluation (IFEval)** benchmark.

   Read the user instruction carefully and produce ONE final response that satisfies *every*
   constraint implied by the instruction IDs and by the instruction text itself.

   ### Rules
   1. Think step-by-step **silently** – do **NOT** reveal your reasoning.
   2. Follow all punctuation, formatting, length, highlighting and stylistic constraints exactly.
   3. If an instruction forbids an element (e.g. no commas), *never* include it.
   4. If an instruction sets a minimum (e.g. ≥ 300 words, ≥ 3 highlighted sections), be sure to exceed it.
   5. Return **only** the finished response text – no explanations, no markdown fences, no extra whitespace.

   Begin now.  Remember: output only the compliant answer.
   """

MMLU_PROMPT = """
   You are a professional evaluation expert. Please answer the following question directly.
   Instructions:
   1. Analyze the problem carefully and provide your step-by-step reasoning within <think>...</think> tags.
   2. Provide only the final answer within <answer>...</answer> tags, ensuring it matches the exact format required by the problem (e.g., 'True', 'False', '(A)', '(B)', a space-separated string, or a sequence of characters).
   3. Do NOT include any explanation, justification, or text outside the <think> and <answer> tags.
   4. Ensure the final answer is a single line with no extra whitespace or formatting.

   Question:
   {question}

   Options:
   {options}

   For mathematical problems, make sure to:
   1. Break down the problem into simpler parts
   2. Solve each part methodically
   3. Check your work and verify your answer
   4. Provide your final answer in a clear format
   <think>
   [Your step-by-step reasoning here]
   </think>
   <answer>
   [Your final answer here]
   </answer>
   """

HUMANEVAL_PROMPT = """"""

MBPP_PROMPT = """"""