import os
import re
import shutil
from datetime import datetime
from typing import Dict

from anthropic_api import get_anthropic_response
from llama import generate_text_from_llama

import tomli
from dotenv import load_dotenv
import argparse
from torchtune.models.llama3 import llama3_tokenizer

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

def get_tokenizer(tokenizer_path: str, model_name: str):
    # check if path is valid
    if not os.path.exists(tokenizer_path):
        raise ValueError(f"Tokenizer path {tokenizer_path} does not exist")
    
    if "llama3" in model_name:
        return llama3_tokenizer(tokenizer_path)
    else:
        raise ValueError(f"Tokenizer for {model_name} not supported")

def route_to_model(model_name: str, prompt: str, checkpoint_files: Optional[str] = None, tokenizer_path: Optional[str] = None):
    if "llama3_2" in model_name:
        if checkpoint_files is None:
            raise ValueError("Checkpoint files are required for llama3_2")
        if tokenizer_path is None:
            raise ValueError("Tokenizer path is required for llama3_2")
        tokenizer = get_tokenizer(tokenizer_path, model_name)
        return generate_text_from_llama(prompt, checkpoint_files, tokenizer=tokenizer, model_name=model_name)
    elif "claude" in model_name:
        return get_anthropic_response(prompt, model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")

def main():
    
    # get model name and checkpoint files from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_files", type=str, required=False)
    parser.add_argument("--tokenizer_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    args = parser.parse_args()

    model_name = args.model_name
    checkpoint_files = args.checkpoint_files
    tokenizer_path = args.tokenizer_path


    # grab first prompt in eval_prompts.toml
    with open("prompts/eval_prompts.toml", "rb") as f:
        all_prompts = tomli.load(f)["prompts"]
    # model_name = "claude-3-5-haiku-20241022"
    if args.output_dir is None:
        experiment_directory_name = (
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{model_name}"
        )
    else:
        experiment_directory_name = args.output_dir
    for prompt_dict in all_prompts:
        if prompt_dict.get("skip", False):
            continue
        template_file = prompt_dict["template_file"]
        name = prompt_dict["name"]
        reference_kernel = prompt_dict["reference_kernel"]
        generated_kernel = route_to_model(model_name, prompt_dict, checkpoint_files, tokenizer_path)

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

if __name__ == "__main__":
    main()
