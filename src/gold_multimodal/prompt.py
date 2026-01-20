VQA_THINKING_PROMPT = "Your task: 1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. 3. No extra information or text outside of these tags."

VQA_PROMPT = "Your task: 1. Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. 2. No extra information or text outside of this tag."

HINT_PROMPT = "Your task: 1. You will get a correct answer for the question. 2. Please provide a hint for the question based on the correct answer. inside <hint>...</hint> tags. 3. No extra information or text outside of these tags. The ground truth answer is {answer}."

HINT_AWARE_VQA_THINKING_PROMPT = "Your task: 1. Read the hint provided in <hint>...</hint> tags. 2. Think through the question step by step but do not explicitly mention the hint, enclose your reasoning process in <think>...</think> tags. 3. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags. 4. No extra information or text outside of these tags. The hint is <hint>{hint}</hint>."

