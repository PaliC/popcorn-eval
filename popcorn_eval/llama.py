import os

import torch
import torchtune.generation
from torchao.quantization.quant_api import int8_weight_only, quantize_
from torchtune.models.llama3 import llama3_tokenizer
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
import datetime

END_OF_HEADER_ID_TOKEN = "<|end_header_id|>"

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
END_OF_HEADER_ID_TOKEN = "<|end_header_id|>"

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
    if model_name in model_name_to_class_llama3_2:
        return model_name_to_class_llama3_2[model_name], 'LLAMA3_2'
    else:
        raise ValueError(f"Model {model_name} not found")
    

def generate_text_from_llama(
    checkpoint_files,
    tokenizer,
    model_name: str,
    prompt_dict: Dict[str, str],
    max_generated_tokens=2024,
    temperature=0.0,
):
    # Parse checkpoint directory from first checkpoint file
    checkpoint_dir = "/".join(checkpoint_files[0].split("/")[:-1])
    if not checkpoint_dir:
        checkpoint_dir = "."

    # assert that all checkpoint files exist and have the same directory
    assert all(f.startswith(checkpoint_dir) for f in checkpoint_files)
    assert all(os.path.exists(f) for f in checkpoint_files)
    model_class, model_type = get_model_class_and_type(model_name)
    get_device("cuda")
    prompt = compose_prompt_for_completion(prompt_dict)
    tokenized_prompt = torch.tensor(tokenizer.encode(prompt)).to("cuda")
    # Set up the checkpointer and load state dict
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=[f.split("/")[-1] for f in checkpoint_files],
        output_dir=checkpoint_dir,
        model_type=model_type,
    )
    torchtune_sd = checkpointer.load_checkpoint()

    model = model_class().to("cuda")
    model.load_state_dict(torchtune_sd["model"])
    quantize_(model, int8_weight_only())

    output = torchtune.generation.generate(
        model=model,
        prompt=tokenized_prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=temperature,
    )
    tokens, logits = output
    print(tokens)
    generated_text = tokenizer.decode(tokens.tolist(), skip_special_tokens=False)
    if generated_text.contains(END_OF_HEADER_ID_TOKEN):
        generated_text = generated_text.split(END_OF_HEADER_ID_TOKEN)[1]
    return generated_text


if __name__ == "__main__":
    prompt = f"""
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>

            Cutting Knowledge Date: December 2023
            Today Date: 29 October 2024
            You are an AI assistant who helps software engineers write triton kernels which is a type of gpu kernel written in python.t<|eot_id|><|start_header_id|>user<|end_header_id|>
            Write a Triton kernel function that performs matrix multiplication on two 1024x1024 matrices A and B, resulting in a 1024x1024 output matrix C, using BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, and BLOCK_SIZE_K = 32; ensure each block in C is calculated from corresponding blocks in A and B with a tiled approach; use tl.load with masking to handle boundaries, accumulate partial sums, and store the result in C; include Python host code to launch the kernel with proper grid/block sizes, and verify correctness by comparing with torch.matmul, ensuring accuracy within 1e-4.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
    tokenizer = torchtune.models.llama3.llama3_tokenizer(
        "/tmp/Llama-3.2-3B/original/tokenizer.model"
    )
    text = generate_text_from_llama(
        checkpoint_files=[
            "/tmp/Llama-3.2-3B/hf_model_0001_0.pt",
            "/tmp/Llama-3.2-3B/hf_model_0002_0.pt",
        ],
        prompt=prompt,
        tokenizer=tokenizer,
    )
    print(text)
