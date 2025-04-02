instruction_template = """You are a SVG image-to-code generator.
Your task is to generate SVG code that reproduces the target image as accurately as possible.
The target image is represented by the image placeholder.
Reward:
- Format: {format_reward}/{format_penalty} (correct/incorrect formatting)
- Similarity: Your generated SVG code will be evaluated against the target image (score ranges from 0 to 1, higher is better)
Please provide your answer in the following format:
<think> ... your reasoning process ... </think><answer> ... your SVG code ... </answer>
"""

init_observation_template = """
[Initial Observation]:
{observation}
generate your svg code.
Your reponse should be in the format of <think>...</think><answer>...</answer>
"""
