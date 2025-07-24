import os
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

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

if __name__ == "__main__":
    # Set the directory of the result files
    parent_dir = Path(__file__).resolve().parent.parent
    target_dir = os.path.join(parent_dir, "training/results")

    json_data = read_json_files(target_dir)
    
    investigated_params = ["ablation"]
    evaluation_params = ["f1"]

    param_list = list(list([json_datum["params"][key] for key in investigated_params]) for json_datum in json_data)
    
    unique_list = []

    for param in param_list:
        if len(unique_list) > 0:
            dup_flag = False
            for idx in range(len(unique_list)):
                if param == unique_list[idx]:
                    dup_flag = True
            if not dup_flag:
                unique_list.append(param)
        else:
            unique_list.append(param)
    
    print(unique_list)
    eval_holder = dict([(unique_idx, dict(([(eval_param, []) for eval_param in evaluation_params]))) for unique_idx in range(len(unique_list))])

    for datum in json_data:
        for unique_idx in range(len(unique_list)):
            if unique_list[unique_idx] == list([datum["params"][key] for key in investigated_params]):
                for eval_key in evaluation_params:
                    eval_holder[unique_idx][eval_key].append(datum[eval_key])

    for unique_idx in range(len(unique_list)):
        print(f"=== The Combination of {unique_list[unique_idx]} Yielded {len(eval_holder[unique_idx][evaluation_params[0]])} Samples ===")
        for key, value in eval_holder[unique_idx].items():
            if len(value) >= 2:
                print(f"     -> {key}: MEAN={mean(value):.3f}, STD={stdev(value):.3f}")
            elif len(value) == 1:
                print(f"     -> {key}: MEAN={value[0]:.3f}, STD=N/A (only one sample)")
            else:
                print(f"     -> {key}: No data")