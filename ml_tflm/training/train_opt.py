# optuna_sweep.py
import optuna
import os
import uuid
import json
from hydra import initialize, compose
from omegaconf import OmegaConf
from ml_tflm.training.train_exp import main as train_main

RESULTS_DIR = "ml_tflm/training/results"

def objective(trial):
    # Generate a unique directory/filename for this trial
    trial_id = str(uuid.uuid4())
    metric_path = os.path.join(RESULTS_DIR, f"{trial_id}.json")

    # Make sure the directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Define overrides, including unique metric save path
    with initialize(config_path="configs", version_base="1.1"):
        overrides = [
            # f"architecture.model.eegnet_args.activation={trial.suggest_categorical('activation', ['relu', 'hard_swish', 'tanh'])}",
            f"optimizer.learning_rate={trial.suggest_float('lr', 1e-4, 1e-2, log=True)}",
            "training.epochs=10",
            "training.save_ckpt=false",
            f"training.k_fold=true",
            f"training.metric_save_dir={metric_path.replace(os.sep, '/')}"  # safer for cross-platform
        ]
        cfg = compose(config_name="config", overrides=overrides)

    # Run training (will dump metrics to metric_path)
    train_main(cfg)

    # Read metrics
    with open(metric_path, "r") as f:
        metrics = json.load(f)

    # Return the F1 score or val_loss depending on tuning goal
    return metrics["val_loss"]

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")  # We're minimizing negative F1
    study.optimize(objective, n_trials=20)
    print("Best trial params:", study.best_trial.params)
    print("Best trial Loss:", study.best_value)  # Convert back to positive
