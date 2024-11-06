import subprocess
import tomli
from pathlib import Path


def main():
    with open("prompts/eval_prompts.toml", "rb") as f:
        prompts_dict = tomli.load(f)["prompts"]

    ncu_commands = []
    generated_code_paths = []
    
    # get all generated code paths in the generated_code directory
    for path in Path("generated_code").glob("**/*.py"):
        generated_code_paths.append(path)

    for prompt in prompts_dict:
        name = prompt["name"]
        
        # code path for ai generated code
        gen_ai_path = f"generated_code/{name}_ai_generated.py"
        # code path for reference code
        ref_path = f"generated_code/{name}_reference.py"

        # check if both paths exist otherwise skip
        if not Path(gen_ai_path).exists() or not Path(ref_path).exists():
            print(f"Skipping {name} because one or both paths do not exist")
            print(f"There should be files called {gen_ai_path} and {ref_path}")
            print("You may need to run generate_kernels.py to create these files")
            continue

        ncu_command_generated = [
            'ncu',  # Nsight Compute executable
            '-k', name,
            '--log-file', f"{name}_ai_generated.ncu-rep",
            'python', gen_ai_path  # Command to run your Triton script
        ]
        ncu_command_reference = [
            'ncu',  # Nsight Compute executable
            '-k', name,
            '--log-file', f"{name}_reference.ncu-rep",
            'python', ref_path  # Command to run your Triton script
        ]
        ncu_commands.append(ncu_command_generated)
        ncu_commands.append(ncu_command_reference)

    # run all ncu commands
    # TODO: run in parallel
    for command in ncu_commands:
        print(f"Running command: {command}")
        subprocess.run(command, check=True, capture_output=True, text=True)

if __name__ == "__main__":
    main()
