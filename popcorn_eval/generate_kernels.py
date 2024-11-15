import argparse
import os
import re
import shutil
from datetime import datetime
from typing import Dict, Optional

import tomli

from anthropic_api import get_anthropic_response
from llama import generate_text_from_llama
from open_ai_api import get_openai_response
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


def route_to_model(
    model_name: str,
    prompt_dict: Dict[str, str],
    checkpoint_files: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
):
    if "llama3" in model_name:
        if checkpoint_files is None:
            raise ValueError("Checkpoint files are required for llama3_2")
        if tokenizer_path is None:
            raise ValueError("Tokenizer path is required for llama3_2")
        tokenizer = get_tokenizer(tokenizer_path, model_name)
        return generate_text_from_llama(
            prompt_dict=prompt_dict,
            checkpoint_files=checkpoint_files,
            tokenizer=tokenizer,
            model_name=model_name,
        )
    elif "claude" in model_name:
        return get_anthropic_response(prompt_dict, model_name)
    elif "gpt" in model_name or "o1" in model_name:
        return get_openai_response(prompt_dict, model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")


def main():

    # get model name and checkpoint files from args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use for generation.",
    )
    parser.add_argument(
        "--checkpoint_files",
        type=str,
        nargs="+",
        required=False,
        help="Checkpoint files to use for generation. Space separated.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=False,
        help="Tokenizer path to use for generation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory to save generated code.",
    )
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
    total = len(all_prompts)
    count = 0
    for prompt_dict in all_prompts:
        count += 1
        print(f"Generating {count} of {total}")
        if prompt_dict.get("skip", False):
            continue
        template_file = prompt_dict["template_file"]
        name = prompt_dict["name"]
        reference_kernel = prompt_dict["reference_kernel"]
        generated_kernel = route_to_model(
            model_name, prompt_dict, checkpoint_files, tokenizer_path
        )
        print(f"Generated kernel for {name}:\n{generated_kernel}")
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

# commands to run:
# CUDA_VISIBLE_DEVICES=7 with-proxy python generate_kernels.py --model_name llama3_2_3b --checkpoint_files /tmp/Llama-3.2-3B/hf_model_0001_0.pt /tmp/Llama-3.2-3B/hf_model_0002_0.pt --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir finetuned_llama_3b
# CUDA_VISIBLE_DEVICES=8 with-proxy python generate_kernels.py --model_name llama3_2_3b --checkpoint_files /tmp/Llama-3.2-3B/model-00001-of-00002.safetensors /tmp/Llama-3.2-3B/model-00002-of-00002.safetensors --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir base_llama_3b
# with-proxy python generate_kernels.py --model_name claude-3-5-haiku-20241022 --output_dir claude_haiku
# CUDA_VISIBLE_DEVICES=0 python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir finetuned_llama3_1_70b_epoch_1 --checkpoint_files /tmp/Meta-Llama-3-70b/hf_model_0001_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0002_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0003_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0004_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0005_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0006_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0007_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0008_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0009_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0010_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0011_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0012_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0013_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0014_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0015_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0016_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0017_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0018_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0019_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0020_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0021_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0022_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0023_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0024_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0025_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0026_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0027_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0028_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0029_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0030_0.pt
# CUDA_VISIBLE_DEVICES=3 python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir finetuned_llama3_1_70b_epoch_2 --checkpoint_files   /tmp/Meta-Llama-3-70b/hf_model_0001_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0002_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0003_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0004_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0005_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0006_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0007_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0008_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0009_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0010_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0011_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0012_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0013_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0014_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0015_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0016_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0017_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0018_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0019_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0020_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0021_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0022_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0023_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0024_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0025_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0026_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0027_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0028_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0029_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0030_1.pt
# CUDA_VISIBLE_DEVICES=1 python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir base_llama3_1_70b --checkpoint_files /tmp/Llama-3.1-70B/model-00001-of-00030.safetensors /tmp/Llama-3.1-70B/model-00002-of-00030.safetensors /tmp/Llama-3.1-70B/model-00003-of-00030.safetensors /tmp/Llama-3.1-70B/model-00004-of-00030.safetensors /tmp/Llama-3.1-70B/model-00005-of-00030.safetensors /tmp/Llama-3.1-70B/model-00006-of-00030.safetensors /tmp/Llama-3.1-70B/model-00007-of-00030.safetensors /tmp/Llama-3.1-70B/model-00008-of-00030.safetensors /tmp/Llama-3.1-70B/model-00009-of-00030.safetensors /tmp/Llama-3.1-70B/model-00010-of-00030.safetensors /tmp/Llama-3.1-70B/model-00011-of-00030.safetensors /tmp/Llama-3.1-70B/model-00012-of-00030.safetensors /tmp/Llama-3.1-70B/model-00013-of-00030.safetensors /tmp/Llama-3.1-70B/model-00014-of-00030.safetensors /tmp/Llama-3.1-70B/model-00015-of-00030.safetensors /tmp/Llama-3.1-70B/model-00016-of-00030.safetensors /tmp/Llama-3.1-70B/model-00017-of-00030.safetensors /tmp/Llama-3.1-70B/model-00018-of-00030.safetensors /tmp/Llama-3.1-70B/model-00019-of-00030.safetensors /tmp/Llama-3.1-70B/model-00020-of-00030.safetensors /tmp/Llama-3.1-70B/model-00021-of-00030.safetensors /tmp/Llama-3.1-70B/model-00022-of-00030.safetensors /tmp/Llama-3.1-70B/model-00023-of-00030.safetensors /tmp/Llama-3.1-70B/model-00024-of-00030.safetensors /tmp/Llama-3.1-70B/model-00025-of-00030.safetensors /tmp/Llama-3.1-70B/model-00026-of-00030.safetensors /tmp/Llama-3.1-70B/model-00027-of-00030.safetensors /tmp/Llama-3.1-70B/model-00028-of-00030.safetensors /tmp/Llama-3.1-70B/model-00029-of-00030.safetensors /tmp/Llama-3.1-70B/model-00030-of-00030.safetensors
# CUDA_VISIBLE_DEVICES=2 python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir finetuned_llama3_1_70b_1_epoch --checkpoint_files /tmp/Meta-Llama-3-70b/recipe_state.pt
