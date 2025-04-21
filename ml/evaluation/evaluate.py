import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix

from ml.interpretability.attention_visuals import (
    plot_attention_weights_by_level,
    plot_multichannel_eeg_with_attention
)
from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.utils.splits import create_stratified_datasets, create_kfold_stratified_datasets
from ml.models.hierarchical_classifier import EEGNetHierarchicalClassifier
from ml.interpretability.feature_projection import project_features, plot_projection
from ml.interpretability.saliency import compute_vanilla_saliency, simplify_saliency_map, plot_saliency_map
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def generate_eeg_attention_overlay(model, dataloader, config, level_index=0, channel_names=None):
    print("🧠 Generating attention overlay on EEG traces...")
    for batch in dataloader:
        x = batch["data"].to(config.device)
        attn_mask = batch["attention_mask"].to(config.device)

        out = model(x, attention_mask=attn_mask, return_attn_weights=True)
        attn_weights = out.get("attention_weights")

        if attn_weights is not None:
            attn_weights = attn_weights.squeeze(0).detach().cpu().numpy()
            eeg = batch["data"].squeeze(0).cpu().numpy()  # [E, C, T]

            output_path = os.path.join(config.output_dir, f"eeg_with_attention_level{level_index + 1}.png")
            plot_multichannel_eeg_with_attention(
                eeg,
                attn_weights,
                channel_names=channel_names,
                epoch_len=eeg.shape[-1],
                output_path=output_path
            )
        break

def generate_saliency_maps(model, dataloader, config, level_index=0, class_names=None):
    print("🔬 Generating saliency maps...")
    channel_names = getattr(dataloader.dataset, "channel_names", [f"Ch {i}" for i in range(21)])

    for batch in dataloader:
        x = batch["data"].to(config.device)         # [1, E, C, T]
        y = batch["labels"].to(config.device)       # [1, 3]
        m = batch["label_mask"].to(config.device)   # [1, 3]

        epoch = x[0, 0].unsqueeze(0).unsqueeze(0)    # [1, 1, C, T]
        label = y[0, level_index].item()
        label_valid = bool(m[0, level_index].item())

        # --- DEBUG ---
        print(f"Level {level_index + 1} → True label: {label}, Valid: {label_valid}")

        # Get full model logits
        output = model(epoch)
        level_logits = output[f"level{level_index + 1}_logits"]
        pred_class_idx = level_logits.argmax(dim=1).item()

        # Choose target class for saliency
        class_idx = label if label_valid else pred_class_idx

        # Compute saliency
        epoch.requires_grad = True
        saliency = compute_vanilla_saliency(model.eegnet, epoch, class_idx=class_idx)
        saliency_map = simplify_saliency_map(saliency, reduce="none", normalize=True)  # [C, T]

        # Plot heatmap
        import seaborn as sns
        import pandas as pd
        timepoints = np.arange(saliency_map.shape[1])
        df = pd.DataFrame(saliency_map, index=channel_names, columns=timepoints)

        plt.figure(figsize=(12, 10))
        sns.heatmap(df, cmap="Reds", cbar_kws={"label": "Saliency"}, xticklabels=20, yticklabels=True)

        true_label = (
            class_names[label] if label_valid and class_names and label < len(class_names)
            else "N/A"
        )
        pred_label = (
            class_names[pred_class_idx] if class_names and pred_class_idx < len(class_names)
            else str(pred_class_idx)
        )

        plt.title(f"Saliency Map - Vanilla Gradient\nTrue: {true_label} | Predicted: {pred_label}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Channels")
        plt.tight_layout()

        output_path = os.path.join(config.output_dir, f"saliency_map_level{level_index + 1}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Saved saliency map to {output_path}")
        break  # only first sample


def generate_feature_projection(model, dataloader, config, level_index, class_labels, class_names, method="umap"):
    """
    Generates a 2D feature projection plot (e.g., UMAP or t-SNE) for a given level of the hierarchy.

    Args:
        model: Trained EEGNetHierarchicalClassifier model.
        dataloader: DataLoader over the evaluation dataset.
        config: Configuration object with output_dir and device.
        level_index: 0 (level1), 1 (level2), or 2 (level3).
        class_labels: List of raw label values for this level (e.g., [0, 1]).
        class_names: List of string names for this level (e.g., ["Neurotypical", "Epileptic"]).
        method: Dimensionality reduction method, e.g., "umap" or "tsne".
    """
    print(f"📉 Generating feature projection for Level {level_index + 1}...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["data"].to(config.device)
            y = batch["labels"].to(config.device)[:, level_index]
            m = batch["label_mask"].to(config.device)[:, level_index].bool()
            attn_mask = batch["attention_mask"].to(config.device)

            out = model(x, attention_mask=attn_mask, return_features=True)
            feats = out.get("features")

            if feats is not None and m.any():
                feats = feats[m].detach().cpu()
                y = y[m].cpu()
                all_features.append(feats)
                all_labels.append(y)

    if not all_features:
        print(f"⚠️ No valid features found for Level {level_index + 1}. Skipping projection.")
        return

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    projected = project_features(features, labels=labels, method=method)
    plot_path = os.path.join(config.output_dir, f"feature_projection_level{level_index + 1}.png")
    label_name_dict = {v: name for v, name in zip(class_labels, class_names)}

    plot_projection(
        projected_features=projected,
        labels=labels,
        label_names=label_name_dict,
        title=f"{method} Feature Projection (Level {level_index + 1})",
        save_path=plot_path
    )

def evaluate(config, model, test_loader, levels, labels_all, class_names_all, include_attention_on_eeg):
    
    # Grab channel names from the dataset if available
    channel_names = getattr(test_loader.dataset, "channel_names", [f"Ch {i}" for i in range(21)])

    # Check if the attention overlay on the EEG should be included 
    if include_attention_on_eeg:
        if config.pooling_type == "attention":
            generate_eeg_attention_overlay(
                model,
                test_loader,
                config,
                level_index=0,
                channel_names=[f"Ch {i}" for i in range(21)]
            )

    # Calculate Saliency maps
    if config.pooling_type in ["mean", "attention", "transformer"]:
        generate_saliency_maps(
            model=model,
            dataloader=test_loader,
            config=config,
            level_index=0,
        )

    # Calculate the feature projections - UMAP or t-SNE
    for level_index in range(len(levels)):
        generate_feature_projection(
            model,
            test_loader,
            config,
            level_index=level_index,
            class_labels=labels_all[level_index],
            class_names=class_names_all[level_index],
            method="umap"  # or "tsne" if you prefer
        )

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

    for i, level in enumerate(levels):
        valid = all_masks[:, i].bool()
        if valid.sum() == 0:
            print(f"⚠️ Skipping {level} — no valid mask entries.")
            continue

        y_true_raw = all_targets[valid, i].cpu().numpy()
        y_pred_raw = all_preds[valid, i].cpu().numpy()

        print(f"📊 {level} raw y_true values: {np.unique(y_true_raw)}")
        print(f"📊 {level} raw y_pred values: {np.unique(y_pred_raw)}")

        label_mapping = {v: j for j, v in enumerate(labels_all[i])}
        y_true = np.array([label_mapping.get(lbl, -1) for lbl in y_true_raw])
        y_pred = y_pred_raw

        valid_indices = (y_true != -1)
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
            plot_attention_weights_by_level(model, test_loader, config, i, labels_all[i], class_names_all[i])

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Optional override arguments")
    args = parser.parse_args()

    base_config = OmegaConf.load(args.config)
    override_config = OmegaConf.from_dotlist(args.overrides or [])
    config = OmegaConf.merge(base_config, override_config)

    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    label_map = {
        'neurotypical': 0,
        'epileptic': 1,
        'focal': 2,
        'generalized': 3,
        'left': 4,
        'right': 5
    }

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

    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

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
    model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt"))["model_state_dict"])
    model.eval().to(config.device)

    evaluate(config, model, test_loader, levels, labels_all, class_names_all, include_attention_on_eeg=False)

if __name__ == "__main__":
    main()
