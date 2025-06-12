BBH_PROMPT = """
- Ensure the final answer is a single line with no extra whitespace or formatting.
- Match the answer format to the problem type, such as:
   - Boolean problems: 'True' or 'False'
   - date_understanding: '(A)', '(B)', '(C)', etc.
   - Multiple-choice problems: '(A)', '(B)', '(C)', etc.
   - Sequence completion problems: A sequence of closing brackets like `)`, `]`, `}`, or `>` (For example, if input is `< < [ ( ) ] >`,then you just need to output `>` to complete the sequence, rather than the whole sequence)
   - Word sorting problems: Space-separated words in alphabetical order
   - Causal judgment or web of lies problems: 'Yes' or 'No'
   - Sports understanding problems: 'Yes' or 'No'
   - Formal fallacies: 'valid' or 'invalid'

<answer>
[Your final answer here]
</answer>
"""

MATH_PROMPT = """
- Check for any calculation errors or logical flaws
- Output the final answer in the format: \boxed{{answer}} without any other text. The final answer directly answers the question.
"""

DROP_PROMPT = """ 
- Your reply **MUST** contain **exactly two** XML-like blocks and nothing else:

   <answer>
   …ONLY the final answer here (no extra words, no units unless they are part of the answer)…
   </answer>

- Remember:
   1. Put **all** reasoning strictly inside <think> … </think>.
   2. The <answer> block must contain only the short answer string required by the question,
      trimmed of leading/trailing spaces.
   3. Output absolutely nothing outside those two blocks.
   """

IFEVAL_PROMPT = """
- Follow all punctuation, formatting, length, highlighting and stylistic constraints exactly.
- If an instruction forbids an element (e.g. no commas), *never* include it.
- If an instruction sets a minimum (e.g. ≥ 300 words, ≥ 3 highlighted sections), be sure to exceed it.
- Return **only** the finished response text – no explanations, no markdown fences, no extra whitespace.

   Begin now.  Remember: output only the compliant answer.
   """

MMLU_PROMPT = """
- Provide only the final answer within <answer>...</answer> tags, ensuring it matches the exact format required by the problem (e.g., 'True', 'False', '(A)', '(B)', a space-separated string, or a sequence of characters).
- Ensure the final answer is a single line with no extra whitespace or formatting.
   <answer>
   [Your final answer here]
   </answer>
   """