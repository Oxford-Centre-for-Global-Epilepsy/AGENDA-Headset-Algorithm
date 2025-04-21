import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.utils.splits import create_stratified_datasets, create_kfold_stratified_datasets
from ml.models.hierarchical_classifier import EEGNetHierarchicalClassifier
from ml.training.metrics import compute_per_level_metrics
from ml.evaluation.attention_visualization import plot_eeg_with_attention
from omegaconf import OmegaConf

def load_model(model_path, config):
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
    )
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()
    model.to(config.device)
    return model

def plot_confusion(cm, cm_raw, class_names, title, output_path):
    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = int(cm_raw[i, j])
            text = f"\n({count})"
            ax.text(j + 0.5, i + 0.65, text, ha='center', va='top', color='black', fontsize=8)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(config_path, overrides=None):
    base_config = OmegaConf.load(config_path)
    override_config = OmegaConf.from_dotlist(overrides or [])
    config = OmegaConf.merge(base_config, override_config)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    data_root = os.environ.get("DATA")
    if data_root is None:
        raise EnvironmentError("❌ The $DATA environment variable is not set.")

    if not os.path.isabs(config.output_dir):
        config.output_dir = os.path.join(data_root, config.output_dir)

    config.data_path = os.path.join(
        data_root,
        "AGENDA-Headset-Algorithm/data/final_processed/",
        config.dataset_name + ".h5"
    )

    print(f"📦 Using dataset: {config.dataset_name}")
    print(f"📂 Data path: {config.data_path}")
    print(f"📁 Output dir: {config.output_dir}")

    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"❌ Dataset file not found: {config.data_path}")

    label_map = {
        'neurotypical': 0,
        'epileptic': 1,
        'focal': 2,
        'generalized': 3,
        'left': 4,
        'right': 5
    }

    if config.k_folds > 1:
        _, val_dataset, _ = create_kfold_stratified_datasets(
            h5_path=config.data_path,
            dataset_name=config.dataset_name,
            label_map=label_map,
            k_folds=config.k_folds,
            fold_index=config.fold_index,
            seed=config.seed
        )
    else:
        _, val_dataset, _ = create_stratified_datasets(
            h5_path=config.data_path,
            dataset_name=config.dataset_name,
            label_map=label_map,
            ratios=(0.7, 0.15, 0.15),
            seed=config.seed
        )

    test_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"❌ Best model not found at: {best_model_path}")

    model = load_model(best_model_path, config)

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["data"].to(config.device)
            y = batch["labels"].to(config.device)
            m = batch["label_mask"].to(config.device)
            attn_mask = batch["attention_mask"].to(config.device)

            out = model(x, attention_mask=attn_mask)
            preds = torch.stack([
                out["level1_logits"].argmax(dim=1),
                out["level2_logits"].argmax(dim=1),
                out["level3_logits"].argmax(dim=1)
            ], dim=1)

            all_preds.append(preds)
            all_targets.append(y)
            all_masks.append(m)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    levels = ["level1", "level2", "level3"]
    class_names_all = [
        ["Neurotypical", "Epileptic"],
        ["Focal", "Generalized"],
        ["Left", "Right"]
    ]
    labels_all = [
        [0, 1],
        [2, 3],
        [4, 5]
    ]

    for i, level in enumerate(levels):
        valid = all_masks[:, i].bool()
        if valid.sum() == 0:
            print(f"⚠️ Skipping {level} — no valid mask entries.")
            continue

        y_true_raw = all_targets[valid, i].cpu().numpy()
        y_pred_raw = all_preds[valid, i].cpu().numpy()

        label_mapping = {v: j for j, v in enumerate(labels_all[i])}
        y_true = np.array([label_mapping.get(lbl, -1) for lbl in y_true_raw])
        y_pred = y_pred_raw  # Already in 0/1

        valid_indices = (y_true != -1) & (y_pred != -1)
        if valid_indices.sum() == 0:
            print(f"⚠️ Skipping {level} — y_true contains none of the expected labels {labels_all[i]}.")
            continue

        y_true = y_true[valid_indices]
        y_pred = y_pred[valid_indices]

        cm_raw = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_normalized = np.nan_to_num(
            cm_raw.astype("float") / cm_raw.sum(axis=1, keepdims=True), nan=0.0
        )

        output_file = os.path.join(config.output_dir, f"confusion_matrix_{level}.png")
        plot_confusion(cm_normalized, cm_raw, class_names_all[i], f"Confusion Matrix - {level.capitalize()}", output_file)
        print(f"✅ Saved {level} confusion matrix to {output_file}")

    if config.pooling_type == "attention":
        for i in range(3):
            plot_attention_weights_by_level(
                model=model,
                dataloader=test_loader,
                config=config,
                level_index=i,
                class_labels=labels_all[i],
                class_names=class_names_all[i]
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args.config, args.overrides)
