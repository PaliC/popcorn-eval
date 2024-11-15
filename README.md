# Popcorn Eval

Popcorn Eval is a tool designed to evaluate llm prompts and models which are meant to generate GPU kernels.
Currently we are using Anthropic's API to generate kernels. However, we plan on adding support for other models in the future including custom fine tuned models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/popcorn_eval.git
   cd popcorn_eval
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables by running `cp popcorn_eval/.env.example popcorn_eval/.env` and adding your Anthropic API key:


## Usage

1. Prepare your prompts in the `popcorn_eval/prompts/eval_prompts.toml` file. Each prompt should include a `system_prompt`, `user_prompt`, and `reference_kernel`. As well as a `template_file` which points to the template file that the prompt is for. The template file should contain a `{{ GENERATED_CODE }}` tag which will be replaced with the generated kernel. It should also contain a `name` tag which is the name of the kernel.

2. Run the kernel generation script:

   ```bash
   cd popcorn_eval
   python generate_kernels.py
   ```

3. The generated kernels will be saved in the `generated_code` directory, with both AI-generated and reference versions.

4. Once the kernels are generated, you can run `cd popcorn_eval && python run_ncu.py` to evaluate the performance of the generated kernels.
- When running `run_ncu.py` everything in the `generated_code` directory will be evaluated will be run through ncu. Similarily, for plot generation, the all experiments (subdirectories in `plots`) will be used for the plots.

5. Analyze the results in the `logs` and `plots` directories.




## TODO
- [ ] Add more prompts related to different tasks (e.g. q/a, code completion, etc.)
- [ ] Add labels to prompts. This should be a list which we can use to annoate the results / prompts
- [ ] Add labeling to results to reflect type of task
- [ ] Refine current prompts to ensure they are reasonable
- [ ] Support Multiple Kernel Launches
- [ ] Be able to support training multiple datasets on a single model to see the impact of the dataset on the performance of the model

### Finetuning

For finetuning we use torchtune. In order to install it follow the instructions found here: https://github.com/pytorch/torchtune#installation

We also use wandb. So make an account, grab an api key, and set it up
```
pip install wandb
wandb login
```

Once you have the `tune` cli up and running (checkable by running `tune ls`), you can finetune some models using the dataset in datasets/ using

```bash
# 70b instruct model
tune download meta-llama/Meta-Llama-3-70B-Instruct --output-dir /tmp/Llama-3.1-70B-Instruct/  --ignore-patterns "original/consolidated*" --hf-token <HF_TOKEN>
tune run --nproc_per_node 8 full_finetune_distributed --config finetuning_configs/llama_3_70b_instruct_finetune_config.yaml

tune download meta-llama/Meta-Llama-3-3B --output-dir /tmp/Llama-3.2-3B/  --ignore-patterns "original/consolidated*" --hf-token <HF_TOKEN>
tune full_finetune_single_device --config finetuning_configs/llama_3_3b_finetune_config.yaml
```

from here we can do the generations for the base models, produced models, and claude using
```bash
# todo: modify command lines args to search for checkpoint files
cd popcorn_eval

python generate_kernels.py --model_name llama3_2_3b --checkpoint_files /tmp/Llama-3.2-3B/hf_model_0001_0.pt /tmp/Llama-3.2-3B/hf_model_0002_0.pt --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir finetuned_llama_3b

python generate_kernels.py --model_name llama3_2_3b --checkpoint_files /tmp/Llama-3.2-3B/model-00001-of-00002.safetensors /tmp/Llama-3.2-3B/model-00002-of-00002.safetensors --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir base_llama_3b

python generate_kernels.py --model_name claude-3-5-haiku-20241022 --output_dir claude_haiku

python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.2-3B/original/tokenizer.model --output_dir finetuned_llama3_1_70b_epoch_1 --checkpoint_files /tmp/Meta-Llama-3-70b-Instruct/hf_model_0001_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0002_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0003_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0004_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0005_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0006_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0007_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0008_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0009_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0010_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0011_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0012_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0013_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0014_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0015_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0016_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0017_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0018_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0019_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0020_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0021_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0022_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0023_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0024_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0025_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0026_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0027_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0028_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0029_0.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0030_0.pt

python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.1-70B-Instruct/original/tokenizer.model --output_dir finetuned_llama3_1_70b_epoch_2 --checkpoint_files   /tmp/Meta-Llama-3-70b-Instruct/hf_model_0001_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0002_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0003_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0004_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0005_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0006_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0007_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0008_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0009_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0010_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0011_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0012_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0013_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0014_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0015_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0016_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0017_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0018_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0019_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0020_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0021_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0022_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0023_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0024_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0025_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0026_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0027_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0028_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0029_1.pt    /tmp/Meta-Llama-3-70b-Instruct/hf_model_0030_1.pt

python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.1-70B-Instruct/original/tokenizer.model --output_dir base_llama3_1_70b --checkpoint_files /tmp/Llama-3.1-70B/model-00001-of-00030.safetensors /tmp/Llama-3.1-70B/model-00002-of-00030.safetensors /tmp/Llama-3.1-70B/model-00003-of-00030.safetensors /tmp/Llama-3.1-70B/model-00004-of-00030.safetensors /tmp/Llama-3.1-70B/model-00005-of-00030.safetensors /tmp/Llama-3.1-70B/model-00006-of-00030.safetensors /tmp/Llama-3.1-70B/model-00007-of-00030.safetensors /tmp/Llama-3.1-70B/model-00008-of-00030.safetensors /tmp/Llama-3.1-70B/model-00009-of-00030.safetensors /tmp/Llama-3.1-70B/model-00010-of-00030.safetensors /tmp/Llama-3.1-70B/model-00011-of-00030.safetensors /tmp/Llama-3.1-70B/model-00012-of-00030.safetensors /tmp/Llama-3.1-70B/model-00013-of-00030.safetensors /tmp/Llama-3.1-70B/model-00014-of-00030.safetensors /tmp/Llama-3.1-70B/model-00015-of-00030.safetensors /tmp/Llama-3.1-70B/model-00016-of-00030.safetensors /tmp/Llama-3.1-70B/model-00017-of-00030.safetensors /tmp/Llama-3.1-70B/model-00018-of-00030.safetensors /tmp/Llama-3.1-70B/model-00019-of-00030.safetensors /tmp/Llama-3.1-70B/model-00020-of-00030.safetensors /tmp/Llama-3.1-70B/model-00021-of-00030.safetensors /tmp/Llama-3.1-70B/model-00022-of-00030.safetensors /tmp/Llama-3.1-70B/model-00023-of-00030.safetensors /tmp/Llama-3.1-70B/model-00024-of-00030.safetensors /tmp/Llama-3.1-70B/model-00025-of-00030.safetensors /tmp/Llama-3.1-70B/model-00026-of-00030.safetensors /tmp/Llama-3.1-70B/model-00027-of-00030.safetensors /tmp/Llama-3.1-70B/model-00028-of-00030.safetensors /tmp/Llama-3.1-70B/model-00029-of-00030.safetensors /tmp/Llama-3.1-70B/model-00030-of-00030.safetensors

python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.1-70B/original/tokenizer.model --output_dir finetuned_llama3_1_70b_epoch_1 --checkpoint_files /tmp/Meta-Llama-3-70b/hf_model_0001_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0002_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0003_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0004_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0005_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0006_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0007_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0008_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0009_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0010_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0011_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0012_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0013_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0014_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0015_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0016_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0017_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0018_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0019_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0020_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0021_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0022_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0023_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0024_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0025_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0026_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0027_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0028_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0029_0.pt    /tmp/Meta-Llama-3-70b/hf_model_0030_0.pt

python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.1-70B/original/tokenizer.model --output_dir finetuned_llama3_1_70b_epoch_2 --checkpoint_files   /tmp/Meta-Llama-3-70b/hf_model_0001_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0002_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0003_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0004_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0005_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0006_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0007_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0008_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0009_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0010_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0011_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0012_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0013_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0014_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0015_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0016_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0017_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0018_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0019_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0020_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0021_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0022_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0023_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0024_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0025_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0026_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0027_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0028_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0029_1.pt    /tmp/Meta-Llama-3-70b/hf_model_0030_1.pt

python generate_kernels.py --model_name llama3_1_70b --tokenizer_path /tmp/Llama-3.1-70B/original/tokenizer.model --output_dir base_llama3_1_70b --checkpoint_files /tmp/Llama-3.1-70B/model-00001-of-00030.safetensors /tmp/Llama-3.1-70B/model-00002-of-00030.safetensors /tmp/Llama-3.1-70B/model-00003-of-00030.safetensors /tmp/Llama-3.1-70B/model-00004-of-00030.safetensors /tmp/Llama-3.1-70B/model-00005-of-00030.safetensors /tmp/Llama-3.1-70B/model-00006-of-00030.safetensors /tmp/Llama-3.1-70B/model-00007-of-00030.safetensors /tmp/Llama-3.1-70B/model-00008-of-00030.safetensors /tmp/Llama-3.1-70B/model-00009-of-00030.safetensors /tmp/Llama-3.1-70B/model-00010-of-00030.safetensors /tmp/Llama-3.1-70B/model-00011-of-00030.safetensors /tmp/Llama-3.1-70B/model-00012-of-00030.safetensors /tmp/Llama-3.1-70B/model-00013-of-00030.safetensors /tmp/Llama-3.1-70B/model-00014-of-00030.safetensors /tmp/Llama-3.1-70B/model-00015-of-00030.safetensors /tmp/Llama-3.1-70B/model-00016-of-00030.safetensors /tmp/Llama-3.1-70B/model-00017-of-00030.safetensors /tmp/Llama-3.1-70B/model-00018-of-00030.safetensors /tmp/Llama-3.1-70B/model-00019-of-00030.safetensors /tmp/Llama-3.1-70B/model-00020-of-00030.safetensors /tmp/Llama-3.1-70B/model-00021-of-00030.safetensors /tmp/Llama-3.1-70B/model-00022-of-00030.safetensors /tmp/Llama-3.1-70B/model-00023-of-00030.safetensors /tmp/Llama-3.1-70B/model-00024-of-00030.safetensors /tmp/Llama-3.1-70B/model-00025-of-00030.safetensors /tmp/Llama-3.1-70B/model-00026-of-00030.safetensors /tmp/Llama-3.1-70B/model-00027-of-00030.safetensors /tmp/Llama-3.1-70B/model-00028-of-00030.safetensors /tmp/Llama-3.1-70B/model-00029-of-00030.safetensors /tmp/Llama-3.1-70B/model-00030-of-00030.safetensors

```

Now we can get the evaluations
```bash
cd popcorn_eval
python run_ncu.py
```

### Running generations

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
