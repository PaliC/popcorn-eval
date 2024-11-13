import torch
import torchtune.generation
from torchtune.training import FullModelHFCheckpointer, ModelType
from torchtune.models.llama3_2 import llama3_2_3b





# Set the right directory and files
checkpoint_dir = '/tmp/Llama-3.2-3B/'
# fix this
checkpoint_files = [
    'hf_model_0001_0.pt',
    'hf_model_0002_0.pt',
]

# Set up the checkpointer and load state dict
checkpointer = FullModelHFCheckpointer(
    checkpoint_dir=checkpoint_dir,
    checkpoint_files=checkpoint_files,
    output_dir=checkpoint_dir,
    model_type=ModelType.LLAMA2
)
torchtune_sd = checkpointer.load_checkpoint()

model = llama3_2_3b()
model.load_state_dict(torchtune_sd)

generated_text = torchtune.generation.generate(
    model=model, 
    prompt = "Hello, how are you?",
    max_generated_tokens=2024,
    temperature=0.0,
    )

print(generated_text)