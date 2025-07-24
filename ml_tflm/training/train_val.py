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
    # Only test ablation
    ablation_options = [None, ["A1", "A2"], ["T3", "T4"], ["Fp1", "Fp2"]]

    best_score = float("inf")
    best_params = None

    for i, ablation in enumerate(ablation_options):
        ablation_override = (
            f"dataset.ablation={ablation}"
            if ablation is not None else "dataset.ablation=null"
        )
        overrides = [ablation_override,]
        params = {
            "ablation": ablation
        }
        score = run_trial(i, overrides, params)
        if score < best_score:
            best_score = score
            best_params = params

    print("\n=== Ablation Sweep Complete ===")
    print("Best parameters:", best_params)
    print("Best validation score:", best_score)
