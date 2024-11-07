# Popcorn Eval

Popcorn Eval is a tool designed to evaluate llm prompts and models which are meant to generate GPU kernels.

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

3. Set up your environment variables by creating a `.env` file in the root directory and adding your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

1. Prepare your prompts in the `prompts/eval_prompts.toml` file. Each prompt should include a `system_prompt`, `user_prompt`, and `reference_kernel`.

2. Run the kernel generation script:

   ```bash
   python generate_kernels.py
   ```

3. The generated kernels will be saved in the `generated_code` directory, with both AI-generated and reference versions.

## File Structure

- `popcorn_eval/generate_kernels.py`: Main script for generating kernels using the Anthropic API.
- `popcorn_eval/prompts/eval_prompts.toml`: Contains the prompts used for kernel generation.
- `popcorn_eval/code_templates/`: Directory containing template files for different operations (e.g., matrix multiplication, addition, softmax).
- `generated_code/`: Directory where generated kernel code is saved.
- `_helper_functions.py`: Contains helper functions used in the generated code.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.