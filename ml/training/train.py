from omegaconf import OmegaConf
import sys
import os
import csv
import torch
import uuid
import random
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.training.metrics import compute_per_level_metrics
from ml.training.losses import HierarchicalLoss
from ml.training.utils import EarlyStopping
from ml.utils.splits import load_split_indices
from ml.models.hierarchical_classifier import EEGNetHierarchicalClassifier

# don't buffer the standard output, i.e. write straight to the file when able
sys.stdout.reconfigure(line_buffering=True)

def parse_config():
    if "--config" not in sys.argv:
        raise ValueError("Please provide a config file using --config path/to/file.yaml")

    config_index = sys.argv.index("--config")
    config_path = sys.argv[config_index + 1]
    overrides = sys.argv[config_index + 2:]

    base_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(base_config, cli_config)

    if not config.get("run_id"):
        config.run_id = generate_run_name(config)

    # Dynamically define output directory
    base_data_dir = os.environ.get("DATA")
    if base_data_dir is None:
        raise EnvironmentError("❌ The DATA environment variable is not set on this system.")

    # Get the path to the dataset to be used
    config.output_dir = os.path.join(
        base_data_dir,
        "AGENDA-Headset-Algorithm/outputs",
        config.experiment_name,
        config.run_id
    )

    return config

def generate_run_name(config):
    parts = [f"fold_{config.dataset.fold_index}"]
    if config.dataset.get("drop_electrodes"):
        omitted = "-".join(sorted(config.dataset.drop_electrodes))
        parts.append(f"omit_{omitted}")
    return "_".join(parts)

def generate_tags(config):
    tags = {
        "experiment_name": config.experiment_name,
        "fold": str(config.dataset.fold_index),
    }
    if config.dataset.get("drop_electrodes"):
        tags["drop_electrodes"] = ",".join(sorted(config.dataset.drop_electrodes))
    return tags

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🔒 Set global random seed to: {seed}")

def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]

def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_bytes = param_size + buffer_size
    return size_all_bytes / 1e6  # Convert to MB

def main():
    config = parse_config()
    os.makedirs(config.output_dir, exist_ok=True)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {config.device}")

    set_seed(config.seed)

    config_save_path = os.path.join(config.output_dir, "config_used.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(config, f)
    print(f"📄 Saved config to: {config_save_path}")

    log_path = os.path.join(config.output_dir, "metrics_log.csv")
    log_fields = [
        "run_id", "fold", "epoch", "train_loss", "val_loss",
        "level1_accuracy", "level1_f1", "level1_precision", "level1_recall", "level1_roc_auc",
        "level2_accuracy", "level2_f1", "level2_precision", "level2_recall", "level2_roc_auc",
        "level3_accuracy", "level3_f1", "level3_precision", "level3_recall", "level3_roc_auc"
    ]

    with open(log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(config.experiment_name)

    run_name = generate_run_name(config)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(generate_tags(config))
        mlflow.log_params({
            "experiment_name": config.experiment_name,
            "dataset_name": config.dataset.dataset_name, 
            "dropped_electrodes": config.dataset.drop_electrodes,
            "fold_index": config.dataset.fold_index,
            "batch_size": config.train.batch_size,
            "epochs": config.train.epochs,
            "learning_rate": config.train.learning_rate,
            "dropout_rate": config.model.dropout_rate,
            "kernel_length": config.model.kernel_length,
            "pooling_type": config.model.pooling_type
        })

        label_map = {
            'neurotypical': 0,
            'epileptic': 1,
            'focal': 2,
            'generalized': 3,
            'left': 4,
            'right': 5
        }

        # Get paths to the folder containing the dataset train/val and test dataset split files
        split_base_path = config.dataset.subject_split_dir

        fold_split_path = os.path.join(split_base_path, f"fold_{config.dataset.fold_index}.json")
        test_split_path = os.path.join(split_base_path, "test_split.json")

        # Load the subject splits
        fold_splits = load_split_indices(fold_split_path)
        test_splits = load_split_indices(test_split_path)

        # Merge them
        subject_splits = {
            "train": fold_splits["train"],
            "val": fold_splits["val"],
            "test": test_splits["test"]
        }

        # Trying to drop electrodes 
        print(f"Trying to Drop Electrodes: {config.dataset.drop_electrodes}")

        train_dataset = EEGRecordingDataset(
            config.dataset.dataset_path, config.dataset.dataset_name, label_map,
            omit_channels=config.dataset.drop_electrodes, 
            subject_ids=subject_splits["train"]
        )

        val_dataset = EEGRecordingDataset(
            config.dataset.dataset_path, config.dataset_name, label_map,
            omit_channels=config.dataset.drop_electrodes, 
            subject_ids=subject_splits["val"]
        )

        test_dataset = EEGRecordingDataset(
            config.dataset.dataset_path, config.dataset_name, label_map,
            omit_channels=config.dataset.drop_electrodes, 
            subject_ids=subject_splits["test"]
        )

        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)

        # Get the number of channels in the dataset, i.e. want to determine dynamically for electrode ablation experiments
        num_channels = train_dataset.get_num_channels() 

        # Print the Number of EEG Channels
        print(f"Using {num_channels} EEG Channels: ")
        print(f"Omitted Channels: {train_dataset.get_omitted_channel_names()}")
        print(f"Using Channels: {train_dataset.get_channel_names()}")

        eegnet_args = {
            "num_channels": num_channels,
            "num_samples": 129,
            "dropout_rate": config.model.dropout_rate,
            "kernel_length": config.model.kernel_length
        }

        pooling_args = {"hidden_dim": 64} if config.model.pooling_type == "attention" else {}
        model = EEGNetHierarchicalClassifier(
            eegnet_args=eegnet_args,
            pooling_type=config.model.pooling_type,
            pooling_args=pooling_args
        ).to(config.device)

        early_stopping = EarlyStopping(patience=config.early_stopping.patience, min_delta=config.early_stopping.min_delta)
        criterion = HierarchicalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)

        start_epoch = 0
        best_val_loss = float("inf")
        final_epoch = config.train.epochs

        if getattr(config.train, "resume", False):
            ckpt_path = config.train.get("checkpoint_path") or os.path.join(config.output_dir, "latest_checkpoint.pt")
            if os.path.exists(ckpt_path):
                print(f"🔁 Resuming from checkpoint: {ckpt_path}")
                start_epoch, best_val_loss = load_checkpoint(ckpt_path, model, optimizer)
                print(f"➡️  Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
            else:
                print(f"⚠️ No checkpoint found at {ckpt_path}, starting from scratch.")

        for epoch in range(start_epoch, config.train.epochs):
            model.train()
            total_loss = 0

            for batch in train_loader:
                x = batch["data"].to(config.device)
                y = batch["labels"].to(config.device)
                m = batch["label_mask"].to(config.device)
                attn_mask = batch["attention_mask"].to(config.device)

                out = model(x, attention_mask=attn_mask)
                loss = criterion(out, y, m)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config.train.epochs} - Train Loss: {avg_loss:.4f}")

            model.eval()
            val_loss = 0
            all_preds, all_targets, all_masks = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    x = batch["data"].to(config.device)
                    y = batch["labels"].to(config.device)
                    m = batch["label_mask"].to(config.device)
                    attn_mask = batch["attention_mask"].to(config.device)

                    out = model(x, attention_mask=attn_mask)
                    loss = criterion(out, y, m)
                    val_loss += loss.item()

                    preds = torch.stack([
                        out["level1_logits"].argmax(dim=1),
                        out["level2_logits"].argmax(dim=1),
                        out["level3_logits"].argmax(dim=1)
                    ], dim=1)

                    all_preds.append(preds)
                    all_targets.append(y)
                    all_masks.append(m)

            val_loss /= len(val_loader)
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_masks = torch.cat(all_masks, dim=0)

            metrics = compute_per_level_metrics(all_preds, all_targets, all_masks)

            print(f"🧪 Validation Loss: {val_loss:.4f}")
            for level, stat in metrics.items():
                acc = stat["accuracy"]
                f1 = stat["f1"]
                precision = stat["precision"]
                recall = stat["recall"]
                roc_auc = stat["roc_auc"]
                print(f"  {level.upper()} - Acc: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC_AUC: {roc_auc:.4f}")

            # Calculate the size of the model
            model_size = get_model_size(model)

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "level1_accuracy": metrics["level1"]["accuracy"],
                "level1_f1": metrics["level1"]["f1"],
                "level2_accuracy": metrics["level2"]["accuracy"],
                "level2_f1": metrics["level2"]["f1"],
                "level3_accuracy": metrics["level3"]["accuracy"],
                "level3_f1": metrics["level3"]["f1"], 
                "model_size": model_size
            }, step=epoch + 1)

            with open(log_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_fields)
                writer.writerow({
                    "run_id": config.run_id,
                    "fold": config.dataset.fold_index,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "level1_accuracy": metrics["level1"]["accuracy"],
                    "level1_f1": metrics["level1"]["f1"],
                    "level2_accuracy": metrics["level2"]["accuracy"],
                    "level2_f1": metrics["level2"]["f1"],
                    "level3_accuracy": metrics["level3"]["accuracy"],
                    "level3_f1": metrics["level3"]["f1"], 
                    "model_size": model_size
                })

            early_stopping(val_loss)
            if early_stopping.should_stop:
                print("⏹️ Early stopping triggered!")
                final_epoch = epoch + 1
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(config.output_dir, "best_model.pt"),
                                model, optimizer, epoch + 1, best_val_loss)

            save_checkpoint(os.path.join(config.output_dir, "latest_checkpoint.pt"),
                            model, optimizer, epoch + 1, best_val_loss)

        mlflow.log_metric("stopped_epoch", final_epoch)
        mlflow.pytorch.log_model(model, "model")

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
