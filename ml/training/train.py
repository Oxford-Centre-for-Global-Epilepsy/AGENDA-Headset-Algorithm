from omegaconf import OmegaConf
import sys
import os
import csv
import torch
from torch.utils.data import DataLoader
from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.training.metrics import compute_per_level_metrics
from ml.training.losses import HierarchicalLoss
from ml.training.utils import EarlyStopping
from ml.utils.splits import create_stratified_datasets, create_kfold_stratified_datasets
from ml.models.hierarchical_classifier import EEGNetHierarchicalClassifier

def parse_config():
    if "--config" not in sys.argv:
        raise ValueError("Please provide a config file using --config path/to/file.yaml")

    config_index = sys.argv.index("--config")
    config_path = sys.argv[config_index + 1]
    overrides = sys.argv[config_index + 2:]

    base_config = OmegaConf.load(config_path)
    cli_config = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(base_config, cli_config)
    return config

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

def main():
    config = parse_config()
    os.makedirs(config.output_dir, exist_ok=True)

    # Add logging for tracking performance
    log_path = os.path.join(config.output_dir, "metrics_log.csv")
    log_fields = [
        "epoch", "train_loss", "val_loss",
        "level1_accuracy", "level1_f1",
        "level2_accuracy", "level2_f1",
        "level3_accuracy", "level3_f1"
    ]

    with open(log_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()

    label_map = {
        'neurotypical': 0,
        'epileptic': 1,
        'focal': 2,
        'generalized': 3,
        'left': 4,
        'right': 5
    }

    if config.k_folds > 1:
        train_dataset, val_dataset, test_dataset = create_kfold_stratified_datasets(
            h5_path=config.data_path,
            dataset_name=config.dataset_name,
            label_map=label_map,
            k_folds=config.k_folds,
            fold_index=config.fold_index,
            seed=42
        )
    else:
        train_dataset, val_dataset, test_dataset = create_stratified_datasets(
            h5_path=config.data_path,
            dataset_name=config.dataset_name,
            label_map=label_map,
            ratios=(0.7, 0.15, 0.15),
            seed=42
        )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    eegnet_args = {
        "num_channels": 21,
        "num_samples": 129,
        "dropout_rate": config.dropout_rate,
        "kernel_length": config.kernel_length
    }

    pooling_args = {"hidden_dim": 64} if config.pooling_type == "attention" else {}
    model = EEGNetHierarchicalClassifier(
        eegnet_args=eegnet_args,
        pooling_type=config.pooling_type,
        pooling_args=pooling_args
    ).to(config.device)

    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    criterion = HierarchicalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    start_epoch = 0
    best_val_loss = float("inf")

    if getattr(config, "resume", False):
        ckpt_path = config.get("checkpoint_path") or os.path.join(config.output_dir, "latest_checkpoint.pt")
        if os.path.exists(ckpt_path):
            print(f"🔁 Resuming from checkpoint: {ckpt_path}")
            start_epoch, best_val_loss = load_checkpoint(ckpt_path, model, optimizer)
            print(f"➡️  Resumed at epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"⚠️ No checkpoint found at {ckpt_path}, starting from scratch.")

    for epoch in range(start_epoch, config.epochs):
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
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_loss:.4f}")

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
            print(f"  {level.upper()} - Acc: {acc:.4f}, F1: {f1:.4f}")

        # Log the metrics for this training step
        with open(log_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "level1_accuracy": metrics["level1"]["accuracy"],
                "level1_f1": metrics["level1"]["f1"],
                "level2_accuracy": metrics["level2"]["accuracy"],
                "level2_f1": metrics["level2"]["f1"],
                "level3_accuracy": metrics["level3"]["accuracy"],
                "level3_f1": metrics["level3"]["f1"],
            })

        early_stopping(val_loss)
        if early_stopping.should_stop:
            print("⏹️ Early stopping triggered!")
            break

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(os.path.join(config.output_dir, "best_model.pt"),
                            model, optimizer, epoch + 1, best_val_loss)

        # Save latest checkpoint
        save_checkpoint(os.path.join(config.output_dir, "latest_checkpoint.pt"),
                        model, optimizer, epoch + 1, best_val_loss)

    print("✅ Training complete!")

if __name__ == "__main__":
    main()
