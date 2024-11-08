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

5. Analyze the results in the `logs` directory.

## File Structure

- `popcorn_eval/generate_kernels.py`: Main script for generating kernels using the Anthropic API.
- `popcorn_eval/prompts/eval_prompts.toml`: Contains the prompts used for kernel generation.
- `popcorn_eval/code_templates/`: Directory containing template files for different operations (e.g., matrix multiplication, addition, softmax).
- `generated_code/`: Directory where generated kernel code is saved.
- `_helper_functions.py`: Contains helper functions used in the generated code.


## TODO

- [ ] Add support for other models / make sure this does not rely on Anthropic's API
- [ ] Add support for finetuned model checkpoints
- [ ] Add more prompts
- [ ] Create a summary for the results produced in `popcorn_eval/logs`
- [ ] Create summary of failed prompts / generations
- [ ] Make log parser work on failures

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
