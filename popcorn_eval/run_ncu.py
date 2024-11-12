import csv
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import tomli


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


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
        # check if the file contains "The application returned an error code in the file" line if so return an empty dictionary
        for line in file:
            if "he application returned an error code" in line:
                return {}

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

# returns a list of ncu commands, log pairs, and compilable generated code paths
def get_ncu_commands_for_generated_code():
    # look in generated_code directory for all the generated code
    suffix_list = ["_reference.py", "_ai_generated.py"]
    paths = []
    ncu_commands = []
    logs = []
    compilable_kernels = set()
    for suffix in suffix_list:
        for path in Path("generated_code").glob(f"**/*{suffix}"):
            paths.append(path)
    
    for path in paths:
        experiment_directory_name = path.parent.name
        kernel_name = path.name.split("_")[-1]
        # clean up kernel name by removing the suffix
        for suffix in suffix_list:
            kernel_name = kernel_name.replace(suffix, "")
        ncu_command = [
            "ncu",
            "-k",
            kernel_name,
            "--csv",
            "--log-file",
            f"logs/{experiment_directory_name}/{kernel_name}.ncu-rep",
            "python",
            path,
        ]
        logs.append(f"logs/{experiment_directory_name}/{kernel_name}.ncu-rep")
        ncu_commands.append(ncu_command)
        ai_generated_suffix = "_ai_generated.py"
        if path.name.endswith(ai_generated_suffix):
            with open(path, "r") as f:
                source_code = f.read()
            try:
                compile(source=source_code, filename=path, mode="exec")
                compilable_kernels.add(kernel_name)
            except(SyntaxError, MemoryError):
                pass

    # sort logs and take pairs as log_pairs
    logs.sort()
    log_pairs = list(zip(logs[::2], logs[1::2]))

    return ncu_commands, log_pairs, compilable_kernels

def main():

    # create logs directory if it doesn't exist
    if os.path.exists("logs") is False:
        os.makedirs("logs")

    # get all generated code paths in the generated_code directory
    ncu_commands, log_pairs, compiling_kernels = get_ncu_commands_for_generated_code()

    # run all ncu commands
    # TODO: run in parallel
    counter = 0
    log_to_cosine_similarity = {}
    for command in ncu_commands:
        counter += 1
        cmd_as_str = " ".join(command)
        log_name = command[5]
        print(f"Running command: {cmd_as_str} \n {counter} / {len(ncu_commands)}")
        try:
            capture = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
            )
            for line in capture.stdout.decode().splitlines():
                if "Cosine similarity" in line:
                    cosine_similarity = line.split(":")[1].strip()
                    cosine_similarity = cosine_similarity.replace(",", "")
                    if is_float(cosine_similarity):
                        log_to_cosine_similarity[log_name] = cosine_similarity
                    else:
                        log_to_cosine_similarity[log_name] = "N/A"

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
    experiment_dict_to_csv_rows = defaultdict(lambda: [])
    for log_pair in log_pairs:
        experiment_directory_name = log_pair[0].split("/")[-2]
        os.makedirs(f"logs/{experiment_directory_name}", exist_ok=True)
        ai_generated_csv = parse_profiler_csv(log_pair[0])
        reference_csv = parse_profiler_csv(log_pair[1])
        kernel_name = log_pair[0].split("/")[-1].split("_ai_generated")[0]

        if len(ai_generated_csv) == 0:
            ai_generated_dict = {}
        else:
            ai_generated_dict = ai_generated_csv["0"]
        print(reference_csv)
        reference_dict = reference_csv["0"]
        if len(ai_generated_dict) > 1 or len(reference_dict) > 1:
            print(f"Skipping {kernel_name} because it has more than one kernel launch")

        for metric_name, metric_info in reference_dict.items():
            reference_value = reference_dict[metric_name][0]
            reference_value = reference_value.replace(",", "")
            if not is_float(reference_value):
                continue
            if metric_name not in ai_generated_dict:
                ai_generated_value = "N/A"
                difference = "N/A"
            else:
                ai_generated_value = ai_generated_dict[metric_name][0]
                # remove commas from the values
                ai_generated_value = ai_generated_value.replace(",", "")
                difference = float(ai_generated_value) - float(reference_value)
            csv_row = [
                kernel_name,
                metric_name,
                ai_generated_value,
                reference_value,
                difference,
                metric_info[1],
            ]
            experiment_dict_to_csv_rows[experiment_directory_name].append(csv_row)

        reference_occupancy = reference_dict["Achieved Occupancy"][0]
        reference_theoretical_occupancy = reference_dict["Theoretical Occupancy"][0]
        reference_scaled_occupancy = float(reference_occupancy) / float(
            reference_theoretical_occupancy
        )

        if (
            "Theoretical Occupancy" not in ai_generated_dict
            or "Achieved Occupancy" not in ai_generated_dict
        ):
            ai_generated_occupancy = "N/A"
            ai_generated_theoretical_occupancy = "N/A"
            ai_generated_scaled_occupancy = "N/A"
            scaled_diff = "N/A"
        else:
            # get scaled occupancy (Achieved Occupancy/Theoretical Occupancy)
            ai_generated_occupancy = ai_generated_dict["Achieved Occupancy"][0]
            ai_generated_theoretical_occupancy = ai_generated_dict[
                "Theoretical Occupancy"
            ][0]
            ai_generated_scaled_occupancy = float(ai_generated_occupancy) / float(
                ai_generated_theoretical_occupancy
            )
            scaled_diff = ai_generated_scaled_occupancy - reference_scaled_occupancy
        csv_row = [
            kernel_name,
            "Achieved of Possible Occupancy",
            ai_generated_scaled_occupancy,
            reference_scaled_occupancy,
            scaled_diff,
            "%",
        ]
        experiment_dict_to_csv_rows[experiment_directory_name].append(csv_row)

        # get cosine similarity
        cosine_similarity_reference = log_to_cosine_similarity[log_pair[1]]
        if log_pair[0] not in log_to_cosine_similarity:
            cosine_similarity_ai_generated = "N/A"
            cosine_similarity_diff = "N/A"
        else:
            cosine_similarity_ai_generated = log_to_cosine_similarity[log_pair[0]]
            cosine_similarity_diff = float(cosine_similarity_ai_generated) - float(
                cosine_similarity_reference
            )
        csv_row = [
            kernel_name,
            "Cosine Similarity",
            cosine_similarity_ai_generated,
            cosine_similarity_reference,
            cosine_similarity_diff,
            "",
        ]
        experiment_dict_to_csv_rows[experiment_directory_name].append(csv_row)

        if kernel_name in compiling_kernels:
            experiment_dict_to_csv_rows[experiment_directory_name].append(
                [
                    kernel_name,
                    "gen-ai-compiles",
                    kernel_name in compiling_kernels,
                    "True",
                    "N/A",
                    "",
                ]
            )

        # write to csvs
        for experiment_directory_name, csv_rows in experiment_dict_to_csv_rows.items():
            with open(f"logs/{experiment_directory_name}/00_ncu_results.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                csv_rows.insert(0, csv_columns)
                writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
