import csv
import os
import subprocess
from pathlib import Path

import tomli


def parse_profiler_csv(file_path):
    """
    Parse a CSV file that contains profiler data and convert it to a nested dictionary.
    The CSV parsing starts after encountering 'Disconnected from process' line.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        dict: A nested dictionary where:
            - First level key is the ID
            - Second level key is the Metric Name
            - Values are tuples of (Metric Value, Metric Unit)
    """
    result = {}

    with open(file_path, "r") as file:
        # Skip until we find "Disconnected from process"
        for line in file:
            if "Disconnected from process" in line:
                break

        # Create CSV reader
        reader = csv.DictReader(file)

        # Process each row
        for row in reader:
            id_val = row["ID"]
            metric_name = row["Metric Name"]
            metric_value = row["Metric Value"]
            metric_unit = row["Metric Unit"]

            # Create nested dictionary structure
            if id_val not in result:
                result[id_val] = {}

            # Store metric information as tuple
            result[id_val][metric_name] = (metric_value, metric_unit)

    return result


def main():
    with open("prompts/eval_prompts.toml", "rb") as f:
        prompts_dict = tomli.load(f)["prompts"]

    # create logs directory if it doesn't exist
    if os.path.exists("logs") is False:
        os.makedirs("logs")

    ncu_commands = []
    generated_code_paths = []
    log_pairs = []

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
            "ncu",  # Nsight Compute executable
            "-k",
            name,
            "--csv",
            "--log-file",
            f"logs/{name}_ai_generated.ncu-rep",
            "python",
            gen_ai_path,  # Command to run your Triton script
        ]
        ncu_command_reference = [
            "ncu",  # Nsight Compute executable
            "-k",
            name,
            "--csv",
            "--log-file",
            f"logs/{name}_reference.ncu-rep",
            "python",
            ref_path,  # Command to run your Triton script
        ]
        ncu_commands.append(ncu_command_generated)
        ncu_commands.append(ncu_command_reference)
        log_pairs.append(
            (f"logs/{name}_ai_generated.ncu-rep", f"logs/{name}_reference.ncu-rep")
        )

    # run all ncu commands
    # TODO: run in parallel
    counter = 0
    for command in ncu_commands:
        counter += 1
        cmd_as_str = " ".join(command)
        print(f"Running command: {cmd_as_str} \n {counter} / {len(ncu_commands)}")
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running {cmd_as_str}")
            print("Error message:", e.stderr)
    csv_rows = []
    csv_columns = [
        "kernel_name",
        "Metric Name",
        "AI generated Metric Value",
        "Reference Metric Value",
        "Difference",
        "Metric Unit",
    ]
    csv_rows.append(csv_columns)
    for log_pair in log_pairs:
        ai_generated_csv = parse_profiler_csv(log_pair[0])
        reference_csv = parse_profiler_csv(log_pair[1])
        kernel_name = log_pair[0].split("_")[0]
        print(f"Processing {kernel_name}")
        for metric_name, metric_info in ai_generated_csv.items():
            if metric_name in reference_csv:
                ai_generated_value = metric_info[0]
                reference_value = reference_csv[metric_name][0]
                difference = float(ai_generated_value) - float(reference_value)
                csv_row = [
                    kernel_name,
                    metric_name,
                    ai_generated_value,
                    reference_value,
                    difference,
                    metric_info[1],
                ]
                csv_rows.append(csv_row)

        # write to csv
        with open("logs/ncu_results.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
