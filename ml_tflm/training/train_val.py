import os
import uuid
import json
import gc
import tensorflow as tf
import itertools
import psutil

from hydra import initialize, compose
from ml_tflm.training.train_exp import main as train_main

RESULTS_DIR = "ml_tflm/training/results"

def print_memory_usage():
    used = psutil.Process().memory_info().rss / (1024 ** 2)
    print(f"[Memory] RAM used: {used:.2f} MB")

def run_trial(trial_number, override_list, param_dict):
    trial_id = str(uuid.uuid4())
    metric_path = os.path.join(RESULTS_DIR, f"{trial_id}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n=== Trial {trial_number} ===")
    print("Overrides:", override_list)

    override_list += [
        "training.epochs=20",
        "training.steps_per_epoch=100",
        "training.save_ckpt=false",
        "training.k_fold=true",
        f"training.metric_save_dir={metric_path.replace(os.sep, '/')}"
    ]

    with initialize(config_path="configs", version_base="1.1"):
        cfg = compose(config_name="config", overrides=override_list)

    print_memory_usage()

    try:
        train_main(cfg)
    except Exception as e:
        print(f"[Trial {trial_number}] Training failed: {e}")
        return float("inf")

    try:
        with open(metric_path, "r") as f:
            metrics = json.load(f)
        f1_score = metrics.get("f1", None)
        if f1_score is None:
            raise ValueError("F1 score not found in metrics.")
        score = -f1_score

        # Append trial info
        metrics["trial_number"] = trial_number
        metrics["f1"] = f1_score
        metrics["params"] = param_dict
        with open(metric_path, "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"[Trial {trial_number}] Failed to load F1 score: {e}")
        return float("inf")

    print(f"[Trial {trial_number}] Finished with f1 = {f1_score:.6f}")
    print_memory_usage()

    tf.keras.backend.clear_session()
    gc.collect()

    return score

if __name__ == "__main__":
    trial_configs = [
        {"pooling_layer": "AveragePooling", "ablation": ["A1","A2","Fz","Pz","Cz"]},
        {"pooling_layer": "AveragePooling", "ablation": None},
        #{"pooling_layer": "SingleAttentionPooling", "ablation": ["A1","A2","Fz","Pz","Cz"]},
        #{"pooling_layer": "AblationPooling", "ablation": ["A1","A2","Fz","Pz","Cz"]},
    ]

    for i, config in enumerate(trial_configs):
        pooling_override = f"component/pooling_layer={config['pooling_layer']}"
        ablation_override = f"dataset.ablation={config['ablation']}"
        overrides = [pooling_override, ablation_override]

        print(f"\n=== Trial {i} ===")
        print(f"Pooling: {config['pooling_layer']}, Ablation: {config['ablation']}")
        run_trial(i, overrides, config)

    print("\n=== All Trials Complete ===")

    # trial_configs = [
    #     {"name": "bp40_tr05", "dataset.h5_path": "ml_tflm/dataset/agenda_data_nw_bp45_tr05/combined_south_africa_monopolar_standard_10_20.h5"},
    #     {"name": "bp50", "dataset.h5_path": "ml_tflm/dataset/agenda_data_nw_bp50/combined_south_africa_monopolar_standard_10_20.h5"},
    #     {"name": "bp50_nt50", "dataset.h5_path": "ml_tflm/dataset/agenda_data_nw_bp50_nt50/combined_south_africa_monopolar_standard_10_20.h5"},
    # ]

    # for i, config in enumerate(trial_configs):
    #     h5_override = f"dataset.h5_path={config['dataset.h5_path']}"
    #     overrides = [h5_override]

    #     print(f"\n=== Trial {i} ({config['name']}) ===")
    #     print(f"Dataset path: {config['dataset.h5_path']}")
    #     run_trial(i, overrides, config)

    # print("\n=== All Trials Complete ===")
