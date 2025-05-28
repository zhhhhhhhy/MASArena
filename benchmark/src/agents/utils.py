BBH_FORMAT_PROMPT = """
You are a highly capable AI assistant tasked with solving a problem from the Big-Bench Hard (BBH) dataset. Your response must be precise, structured, and adhere to the expected answer format for the problem type.

Instructions:
1. Analyze the problem carefully and provide your step-by-step reasoning within <think>...</think> tags.
2. Provide only the final answer within <answer>...</answer> tags, ensuring it matches the exact format required by the problem (e.g., 'True', 'False', '(A)', '(B)', a space-separated string, or a sequence of characters).
3. Do NOT include any explanation, justification, or text outside the <think> and <answer> tags.
4. Ensure the final answer is a single line with no extra whitespace or formatting.
5. Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like '> ) }}'
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'

Your response must follow this format:
<think>
[Your step-by-step reasoning here]
</think>
<answer>
[Your final answer here]
</answer>
"""

MMLU_prompt = """You are a professional evaluation expert. Please answer the following question directly.
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

math_prompt = """"""
