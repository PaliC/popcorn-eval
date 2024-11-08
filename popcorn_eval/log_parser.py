import re
from typing import Dict, Optional


def parse_gpu_sol_metrics(filepath: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Parses a file to find and extract the 'Section: GPU Speed Of Light Throughput' table
    and returns it as a dictionary.

    Args:
        filepath (str): Path to the file containing the GPU metrics

    Returns:
        Optional[Dict[str, Dict[str, float]]]: Dictionary containing the parsed metrics,
        or None if table not found. The structure is:
        {
            'metric_name': {
                'value': float,
                'unit': str
            }
        }

    Example:
        >>> metrics = parse_gpu_sol_metrics("profile.ncu-rep")
        >>> if metrics:
        >>>     print(metrics['Memory Throughput']['value'])  # Prints: 39.99
    """

    try:
        with open(filepath, "r") as file:
            content = file.read()

        # Find the section containing the GPU Speed Of Light Throughput table
        pattern = r"Section: GPU Speed Of Light Throughput.*?-{23} -{13} -{12}\n(.*?)\n-{23} -{13} -{12}"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            print("GPU Speed Of Light Throughput table not found in file")
            return None

        # Extract the table content
        table_content = match.group(1).strip()

        # Initialize the results dictionary
        metrics_dict = {}

        # Process the table rows
        for line in table_content.split("\n"):
            # Skip the header row
            if "Metric Name" in line:
                continue

            # Split the line and clean up whitespace
            parts = [part.strip() for part in line.split("  ") if part.strip()]
            if len(parts) == 3:
                metric_name = parts[0]
                metric_unit = parts[1]
                # Clean up the metric value (remove commas and convert to float)
                metric_value = float(parts[2].replace(",", ""))

                metrics_dict[metric_name] = {"value": metric_value, "unit": metric_unit}

        return metrics_dict

    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        return None


if __name__ == "__main__":
    filepath = "logs/softmax_kernel_reference.ncu-rep"
    metrics = parse_gpu_sol_metrics(filepath)
    print(metrics)
    if metrics:
        print("Parsed GPU Speed Of Light Throughput metrics:")
        for metric_name, metric_info in metrics.items():
            print(f"{metric_name}: {metric_info['value']} {metric_info['unit']}")
    else:
        print("No metrics found.")
