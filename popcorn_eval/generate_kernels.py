import os
import re
import shutil
from datetime import datetime
from typing import Dict

import anthropic

import tomli
from dotenv import load_dotenv


def get_anthropic_response(
    prompt: Dict[str, str], model_name="claude-3-5-haiku-20241022"
) -> str:
    """Get response from Anthropic API using Claude model"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = compose_prompt_for_completion(prompt)
    system_prompt = prompt["system_prompt"]
    user_prompt = prompt["user_prompt"]

    # Get completion from Claude
    message = client.messages.create(
        model=model_name,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text


# get todays date in the format Day Month Year
TODAY_DATE = datetime.now().strftime("%d %B %Y")
CUTOFF_KNOWLEDGE_DATE = "December 2023"
SYSTEM_PROMPT_TOKEN = "[[SYSTEM_PROMPT]]"
USER_PROMPT_TOKEN = "[[USER_PROMPT]]"

COMPLETION_PROMPT_TEMPLATE = f"""

  <|begin_of_text|><|start_header_id|>system<|end_header_id|>

  Cutting Knowledge Date: {CUTOFF_KNOWLEDGE_DATE}
  Today Date: {TODAY_DATE}

  {SYSTEM_PROMPT_TOKEN}
  <|eot_id|><|start_header_id|>user<|end_header_id|>

  {USER_PROMPT_TOKEN}
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


def compose_prompt_for_completion(prompt_dict: Dict[str, str]) -> str:
    system_prompt = prompt_dict["system_prompt"]
    user_prompt = prompt_dict["user_prompt"]

    prompt = COMPLETION_PROMPT_TEMPLATE.replace(
        SYSTEM_PROMPT_TOKEN, system_prompt
    ).replace(USER_PROMPT_TOKEN, user_prompt)
    return prompt


def extract_python_code(text: str) -> str:
    # extract python code from string
    # code is of the form ```python ... ```
    # only extract the first code block

    # remove the first and last line
    try:
        python_code = re.search(r"```python\n((.*\n)*)```", text).group(1)
    except AttributeError:
        print(f"Could not find python code in {text}")
        # if no match, return an exception raised with the message could not find python code
        return 'raise Exception("This file was not generated with valid python code")'
    return python_code


if __name__ == "__main__":
    # grab first prompt in eval_prompts.toml
    with open("prompts/eval_prompts.toml", "rb") as f:
        all_prompts = tomli.load(f)["prompts"]
    model_name = "claude-3-5-haiku-20241022"
    experiment_directory_name = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}"
    )
    for prompt_dict in all_prompts:
        if prompt_dict.get("skip", False):
            continue
        template_file = prompt_dict["template_file"]
        name = prompt_dict["name"]
        reference_kernel = prompt_dict["reference_kernel"]
        generated_kernel = get_anthropic_response(prompt_dict, model_name=model_name)

        # parse out python code from generated_kernel
        generated_triton_kernel = extract_python_code(generated_kernel)
        reference_triton_kernel = extract_python_code(reference_kernel)

        # replace {{ GENERATED CODE }} with generated_kernel in the template file
        with open(template_file, "r") as f:
            template_code = f.read()

        template_code_generated = template_code.replace(
            "{{ GENERATED CODE }}", generated_triton_kernel
        )
        template_code_reference = template_code.replace(
            "{{ GENERATED CODE }}", reference_triton_kernel
        )
        # experiment directory name should be the date and time of the experiment + model name
        # create experiment directory
        os.makedirs(f"generated_code/{experiment_directory_name}", exist_ok=True)

        # write to file
        with open(
            f"generated_code/{experiment_directory_name}/{name}_ai_generated.py", "w"
        ) as f:
            f.write(template_code_generated)
        with open(
            f"generated_code/{experiment_directory_name}/{name}_reference.py", "w"
        ) as f:
            f.write(template_code_reference)

    # copy over _helper_functions.py to generated_code
    shutil.copy(
        "code_templates/_helper_functions.py",
        f"generated_code/{experiment_directory_name}",
    )
