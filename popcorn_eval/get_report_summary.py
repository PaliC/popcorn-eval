import json
import os
import pprint
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_experiment_results() -> Dict[str, pd.DataFrame]:
    """
    Scans the logs directory for _00_ncu_results.csv files and creates a mapping
    from experiment name to the contents of its results file.

    Returns:
        Dict mapping experiment name to pandas DataFrame of results
    """
    results = {}
    logs_dir = "logs"

    for experiment in os.listdir(logs_dir):
        experiment_dir = os.path.join(logs_dir, experiment)
        if os.path.isdir(experiment_dir):
            results_file = os.path.join(experiment_dir, "_00_ncu_results.csv")
            if os.path.exists(results_file):
                results[experiment] = pd.read_csv(results_file)
                # for entries in results[experiment] if value is "True" then convert to 1 and if "False" then 0
                results[experiment].replace({"True": 1.0, "False": 0.0}, inplace=True)
                # do the same for bools
                results[experiment].replace({True: 1.0, False: 0.0}, inplace=True)
    return results


def get_report_summary(
    results: Dict[str, pd.DataFrame], metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Generates a summary of the results for each experiment.
    Assume the header is  kernel_name Metric Name AI generated Metric Value Reference Metric Value  Difference Metric Unit
    """
    report_summary_numerical = defaultdict(lambda: defaultdict(float))
    report_summary_success_rate = defaultdict(lambda: defaultdict(float))
    if metrics is None:
        # get all metrics names which is the first column
        first_experiment = list(results.keys())[0]
        metrics = results[first_experiment]["Metric Name"].unique().tolist()
    for metric in metrics:
        # go through all dataframes and get average of the metric if it exists. If a bool true is 1 and false is 0 then average is the percentage of true
        for experiment, df in results.items():
            # special case for compiles as there is no difference column
            if metric == "compiles":
                metric_values = df[df["Metric Name"] == metric][
                    "AI generated Metric Value"
                ].values
            else:
                metric_values = df[df["Metric Name"] == metric]["Difference"].values
            # cast to float64
            metric_values = metric_values.astype(np.float64)
            metric_values_cleaned = metric_values[~np.isnan(metric_values)]
            # if empty then set to 0
            if len(metric_values_cleaned) == 0:
                average_metric_value = 0
            else:
                average_metric_value = np.mean(metric_values_cleaned)
            report_summary_numerical[metric][experiment] = average_metric_value
            success_rate = len(metric_values_cleaned) / len(metric_values)
            report_summary_success_rate[metric][experiment] = success_rate

    return report_summary_numerical, report_summary_success_rate


def create_plots(
    report_summary_numerical: Dict[str, Dict[str, float]],
    report_summary_success_rate: Dict[str, Dict[str, float]],
    output_dir: str,
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # create a bar plot for the numerical summary. Each metric should have a barplot
    # with the experiment names on the x axis and the metric values on the y axis
    for metric, values in report_summary_numerical.items():
        plt.figure(figsize=(12, 8))
        title = f"Difference in {metric} against reference"
        if metric == "compiles":
            title = f"Percentage of generated kernels that"
        sns.barplot(x=list(values.keys()), y=list(values.values())).set_title(title)
        plt.xticks(fontsize=16, rotation=45, ha="right")
        plt.tight_layout()
        metric_cleaned = metric.replace("/", "_")
        plt.savefig(os.path.join(output_dir, f"{metric_cleaned}_numerical_summary.png"))
        plt.close()
    # create a bar plot for the success rate summary. Each metric should have a barplot
    # with the experiment names on the x axis and the success rate on the y axis
    for metric, values in report_summary_success_rate.items():
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(values.keys()), y=list(values.values())).set_title(
            f"{metric} success rate"
        )
        plt.xticks(fontsize=16, rotation=45, ha="right")
        plt.tight_layout()
        metric_cleaned = metric.replace("/", "_")
        plt.savefig(
            os.path.join(output_dir, f"{metric_cleaned}_success_rate_summary.png")
        )
        plt.close()


def generate_plots(output_dir: str) -> None:
    results = get_experiment_results()
    report_summary_numerical, report_summary_success_rate = get_report_summary(results)
    create_plots(report_summary_numerical, report_summary_success_rate, output_dir)


if __name__ == "__main__":
    results = get_experiment_results()
    report_summary_numerical, report_summary_success_rate = get_report_summary(
        results, metrics=["compiles", "Achieved of Possible Occupancy"]
    )
    create_plots(report_summary_numerical, report_summary_success_rate, "plots")
