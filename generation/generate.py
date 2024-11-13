import torch
import torchtune.generation
from torchtune.training import FullModelHFCheckpointer, ModelType
from torchtune.models.llama3_2 import llama3_2_3b
import os




def generate_text(checkpoint_files, prompt="Hello, how are you?", model_type=ModelType.LLAMA3_2, max_generated_tokens=2024, temperature=0.0):
    # Parse checkpoint directory from first checkpoint file
    checkpoint_dir = '/'.join(checkpoint_files[0].split('/')[:-1])
    if not checkpoint_dir:
        checkpoint_dir = '.'

    # assert that all checkpoint files exist and have the same directory
    assert all(f.startswith(checkpoint_dir) for f in checkpoint_files)
    assert all(os.path.exists(f) for f in checkpoint_files)


    # Set up the checkpointer and load state dict
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=checkpoint_dir,
        checkpoint_files=[f.split('/')[-1] for f in checkpoint_files],
        output_dir=checkpoint_dir,
        model_type=model_type
    )
    torchtune_sd = checkpointer.load_checkpoint()

    model = llama3_2_3b()
    model.load_state_dict(torchtune_sd)

    generated_text = torchtune.generation.generate(
        model=model,
        prompt=prompt,
        max_generated_tokens=max_generated_tokens,
        temperature=temperature,
    )

    return generated_text

if __name__ == "__main__":
    generate_text(checkpoint_files=["/tmp/Llama-3.2-3B/hf_model_0001_0.pt", "/tmp/Llama-3.2-3B/hf_model_0002_0.pt"], prompt="Hello, how are you?")