import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import mlflow

def plot_and_log_curves(metrics_csv_path, run_id=None):
    df = pd.read_csv(metrics_csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss Over Epochs")
    ax.legend()
    fig.tight_layout()

    # Save figure locally
    fig_path = os.path.join(os.path.dirname(metrics_csv_path), "loss_curve.png")
    plt.savefig(fig_path)
    print(f"📊 Saved loss plot to: {fig_path}")

    # Log to MLflow
    if run_id:
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(fig_path)
            print(f"📤 Logged plot to MLflow under run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to metrics_log.csv")
    parser.add_argument("--run_id", help="MLflow run ID (optional)")
    args = parser.parse_args()

    plot_and_log_curves(args.csv, args.run_id)
