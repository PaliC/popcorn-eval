import datetime
import os
from typing import Callable, Dict, Tuple

import torch
import torchtune.generation
from torchao.quantization.quant_api import int8_weight_only, quantize_
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.models.llama3._tokenizer import SPECIAL_TOKENS
from torchtune.models.llama3_1 import (
    llama3_1_405b,
    llama3_1_70b,
    llama3_1_8b,
    lora_llama3_1_405b,
    lora_llama3_1_70b,
    lora_llama3_1_8b,
    qlora_llama3_1_405b,
    qlora_llama3_1_70b,
    qlora_llama3_1_8b,
)
from torchtune.models.llama3_2 import (
    llama3_2,
    llama3_2_1b,
    llama3_2_3b,
    lora_llama3_2,
    lora_llama3_2_1b,
    lora_llama3_2_3b,
    qlora_llama3_2_1b,
    qlora_llama3_2_3b,
)

from torchtune.training import FullModelHFCheckpointer, ModelType
from torchtune.utils import get_device

END_OF_HEADER_ID_TOKEN = "<|end_header_id|>"

# get todays date in the format Day Month Year
TODAY_DATE = datetime.datetime.now().strftime("%d %B %Y")
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
END_OF_HEADER_ID_TOKEN = "<|end_header_id|>"


def parse_out_assistant_response(text: str) -> str:
    """
    Parses out the assistant response from the generated text.
    """
    # get all text between first occurence of <|start_header_id|>assistant<|end_header_id|> and <|eot_id|>
    ASSISSTANT_RESPONSE_TOKEN = "<|start_header_id|>assistant<|end_header_id|>"
    EOT_TOKEN = "<|eot_id|>"
    if ASSISSTANT_RESPONSE_TOKEN not in text:
        return ""
    text = text.split(ASSISSTANT_RESPONSE_TOKEN)[1]
    if EOT_TOKEN not in text:
        return text
    return text.split(EOT_TOKEN)[0]


def compose_prompt_for_completion(prompt_dict: Dict[str, str]) -> str:
    system_prompt = prompt_dict["system_prompt"]
    user_prompt = prompt_dict["user_prompt"]

    prompt = COMPLETION_PROMPT_TEMPLATE.replace(
        SYSTEM_PROMPT_TOKEN, system_prompt
    ).replace(USER_PROMPT_TOKEN, user_prompt)
    return prompt


# given a string referencing a model name, it grabs the model class and model_type
def get_model_class_and_type(model_name: str) -> Tuple[Callable, ModelType]:
    model_name_to_class_llama3_2 = {
        "llama3_2": llama3_2,
        "llama3_2_1b": llama3_2_1b,
        "llama3_2_3b": llama3_2_3b,
        "lora_llama3_2": lora_llama3_2,
        "lora_llama3_2_1b": lora_llama3_2_1b,
        "lora_llama3_2_3b": lora_llama3_2_3b,
        "qlora_llama3_2_1b": qlora_llama3_2_1b,
        "qlora_llama3_2_3b": qlora_llama3_2_3b,
    }
    model_name_to_class_llama3_1 = {
        "llama3_1_70b": llama3_1_70b,
        "llama3_1_8b": llama3_1_8b,
        "llama3_1_405b": llama3_1_405b,
        "lora_llama3_1_70b": lora_llama3_1_70b,
        "lora_llama3_1_8b": lora_llama3_1_8b,
        "lora_llama3_1_405b": lora_llama3_1_405b,
        "qlora_llama3_1_70b": qlora_llama3_1_70b,
        "qlora_llama3_1_8b": qlora_llama3_1_8b,
        "qlora_llama3_1_405b": qlora_llama3_1_405b,
    }
    if model_name in model_name_to_class_llama3_2:
        return model_name_to_class_llama3_2[model_name], "LLAMA3_2"
    elif model_name in model_name_to_class_llama3_1:
        return model_name_to_class_llama3_1[model_name], "LLAMA3"
    else:
        raise ValueError(f"Model {model_name} not found")


def generate_text_from_llama(
    checkpoint_files,
    tokenizer,
    model_name: str,
    prompt_dict: Dict[str, str],
    max_generated_tokens=1024,
    temperature=0.6,
):
    # Parse checkpoint directory from first checkpoint file
    checkpoint_dir = "/".join(checkpoint_files[0].split("/")[:-1])
    if not checkpoint_dir:
        checkpoint_dir = "."

    # assert that all checkpoint files exist and have the same directory
    assert all(f.startswith(checkpoint_dir) for f in checkpoint_files)
    assert all(os.path.exists(f) for f in checkpoint_files)
    model_class, model_type = get_model_class_and_type(model_name)
    device = get_device("cuda")
    prompt = compose_prompt_for_completion(prompt_dict)
    tokenized_prompt = torch.tensor(
        tokenizer.encode(prompt, add_bos=False, add_eos=False)
    ).to("cuda")
    # Set up the checkpointer and load state dict
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=[f.split("/")[-1] for f in checkpoint_files],
        output_dir=checkpoint_dir,
        model_type=model_type,
    )
    torchtune_sd = checkpointer.load_checkpoint()

    model = model_class()
    model.load_state_dict(torchtune_sd["model"])
    quantize_(model, int8_weight_only())
    model.to("cuda")

    output = torchtune.generation.generate(
        model=model,
        prompt=tokenized_prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=temperature,
        stop_tokens=[SPECIAL_TOKENS["<|eot_id|>"]],
    )
    tokens, logits = output
    generated_text = tokenizer.decode(tokens.tolist()[0], skip_special_tokens=False)
    generated_text = parse_out_assistant_response(generated_text)
    print(generated_text)
    return generated_text


if __name__ == "__main__":
    prompt_dict = {
        "system_prompt": f"""
            Cutting Knowledge Date: December 2023
            Today Date: 29 October 2024
            You are an AI assistant who helps software engineers write triton kernels which is a type of gpu kernel written in python.
            """,
        "user_prompt": f"""
            Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.
        """,
    }
    tokenizer = torchtune.models.llama3.llama3_tokenizer(
        "/tmp/Llama-3.2-3B/original/tokenizer.model"
    )
    text = generate_text_from_llama(
        checkpoint_files=[
            "/tmp/Llama-3.2-3B/hf_model_0001_0.pt",
            "/tmp/Llama-3.2-3B/hf_model_0002_0.pt",
        ],
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        model_name="llama3_2_3b",
    )
    print(text)
