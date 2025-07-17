import optuna
import os
import uuid
import json
import gc
import tensorflow as tf

import psutil

from hydra import initialize, compose
from ml_tflm.training.train_exp import main as train_main

RESULTS_DIR = "ml_tflm/training/results"

def print_memory_usage():
    used = psutil.Process().memory_info().rss / (1024 ** 2)
    print(f"[Memory] RAM used: {used:.2f} MB")

def objective(trial):
    trial_id = str(uuid.uuid4())
    metric_path = os.path.join(RESULTS_DIR, f"{trial_id}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with initialize(config_path="configs", version_base="1.1"):
        overrides = [
            f"component.feature_extractor.F1={trial.suggest_categorical('F1', [16, 32, 64])}",
            f"component.feature_extractor.F2={trial.suggest_categorical('F2', [4, 8, 12, 16])}",
            f"component.pooling_layer.hidden_dim={trial.suggest_categorical('hidden_dim', [16, 32, 64])}",
            "training.epochs=20",
            "training.save_ckpt=false",
            "training.k_fold=false",
            f"training.metric_save_dir={metric_path.replace(os.sep, '/')}"
        ]
        cfg = compose(config_name="config", overrides=overrides)

    print(f"\n=== Trial {trial.number} starting ===")
    print_memory_usage()

    try:
        train_main(cfg)
    except Exception as e:
        print(f"[Trial {trial.number}] Training failed: {e}")
        return float("inf")

    try:
        with open(metric_path, "r") as f:
            metrics = json.load(f)
        f1_score = metrics.get("f1", None)
        if f1_score is None:
            raise ValueError("F1 score not found in metrics.")
        score = -f1_score  # Because Optuna minimizes by default

        # Append trial info to the metrics file
        metrics["trial_number"] = trial.number
        metrics["f1"] = f1_score  # Redundant in case it wasn't saved
        metrics["params"] = trial.params
        with open(metric_path, "w") as f:
            json.dump(metrics, f, indent=2)

    except Exception as e:
        print(f"[Trial {trial.number}] Failed to load F1 score: {e}")
        return float("inf")

    print(f"[Trial {trial.number}] Finished with f1 = {f1_score:.6f}")
    print_memory_usage()

    # Clean up TensorFlow memory
    tf.keras.backend.clear_session()
    gc.collect()

    return score

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial params:", study.best_trial.params)
    print("Best trial val_loss:", study.best_value)
