import os
import json
from pathlib import Path
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def read_json_files(directory):
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                try:
                    data = json.load(file)
                    json_data.append(data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")
    return json_data


def summarize_results_by_param_combo(json_data, investigated_params, evaluation_params):
    assert len(investigated_params) >= 1, "You must specify at least one parameter to investigate."

    param_list = [
        [datum["params"][key] for key in investigated_params]
        for datum in json_data
    ]

    unique_list = []
    for param in param_list:
        if param not in unique_list:
            unique_list.append(param)

    eval_holder = {
        idx: {key: [] for key in evaluation_params}
        for idx in range(len(unique_list))
    }

    for datum in json_data:
        param_combo = [datum["params"][key] for key in investigated_params]
        for idx, unique_param in enumerate(unique_list):
            if param_combo == unique_param:
                for eval_key in evaluation_params:
                    eval_holder[idx][eval_key].append(datum[eval_key])
                break

    for idx, param_combo in enumerate(unique_list):
        print(f"=== The Combination of {param_combo} Yielded {len(eval_holder[idx][evaluation_params[0]])} Samples ===")
        for key, values in eval_holder[idx].items():
            if len(values) >= 2:
                print(f"     -> {key}: MEAN={mean(values):.3f}, STD={stdev(values):.3f}")
            elif len(values) == 1:
                print(f"     -> {key}: MEAN={values[0]:.3f}, STD=N/A (only one sample)")
            else:
                print(f"     -> {key}: No data")

def plot_violin_by_numeric_param(json_data, investigated_param, evaluation_param, bin_width=8):
    param_vals = []
    eval_vals = []

    for datum in json_data:
        try:
            x = datum["params"][investigated_param]
            y = datum[evaluation_param]
            param_vals.append(x)
            eval_vals.append(y)
        except KeyError:
            continue

    if not param_vals:
        print(f"No data found for parameter '{investigated_param}' and metric '{evaluation_param}'")
        return

    if bin_width == 1:
        # Skip binning
        x_vals = param_vals
        x_labels = sorted(set(x_vals))
        df = pd.DataFrame({
            investigated_param: x_vals,
            evaluation_param: eval_vals
        })
        order = x_labels
        xlabel = investigated_param
        title = f"{evaluation_param} by {investigated_param}"
    else:
        # Bin the numeric param
        binned_param_vals = [
            f"{(x // bin_width) * bin_width}-{(x // bin_width) * bin_width + bin_width - 1}"
            for x in param_vals
        ]

        # Extract bin sort keys for correct plotting order
        bin_sort_keys = {
            bin_label: int(bin_label.split("-")[0])
            for bin_label in binned_param_vals
        }

        df = pd.DataFrame({
            investigated_param: binned_param_vals,
            evaluation_param: eval_vals
        })

        order = sorted(set(binned_param_vals), key=lambda b: bin_sort_keys[b])
        xlabel = f"{investigated_param} (binned by {bin_width})"
        title = f"{evaluation_param} by binned {investigated_param}"

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=investigated_param, y=evaluation_param, data=df, order=order)
    plt.xlabel(xlabel)
    plt.ylabel(evaluation_param)
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Locate result files
    parent_dir = Path(__file__).resolve().parent.parent
    target_dir = os.path.join(parent_dir, "training", "results")

    # Load JSON trials
    json_data = read_json_files(target_dir)

    # Text summary for param combos
    summarize_results_by_param_combo(
        json_data,
        investigated_params=["pooling_layer"],
        evaluation_params=["f1"]
    )

    # Plot violin for a single numeric param
    # plot_violin_by_numeric_param(
    #     json_data,
    #     investigated_param="F1",
    #     evaluation_param="f1",
    #     bin_width=4
    # )
