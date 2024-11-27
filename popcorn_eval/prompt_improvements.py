import json
import random
from typing import List


def add_triton_examples(prompt: str, num_examples_to_add: int) -> str:
    golden_json = "datasets/popular_triton.json"
    all_examples = json.load(open(golden_json))
    chosen_examples = []
    commented_examples = [
        example["input"] for example in all_examples if '"""' in example["input"]
    ]
    uncommented_examples = [
        example["input"] for example in all_examples if '"""' not in example["input"]
    ]

    # shuffle the lists
    random.shuffle(commented_examples)
    random.shuffle(uncommented_examples)
    if len(commented_examples) < num_examples_to_add:
        chosen_examples.extend(commented_examples[:num_examples_to_add])
    else:
        chosen_examples.extend(commented_examples[:num_examples_to_add])
        chosen_examples.extend(
            uncommented_examples[: num_examples_to_add - len(commented_examples)]
        )
    prompt += "\n\nHere are some Examples of well written triton functions:\n\n"
    for example in chosen_examples:
        prompt += f"{example}\n\n"
    return prompt
