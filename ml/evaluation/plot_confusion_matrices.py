import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.utils.splits import create_stratified_datasets, create_kfold_stratified_datasets
from ml.models.hierarchical_classifier import EEGNetHierarchicalClassifier
from ml.training.metrics import compute_per_level_metrics
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


def plot_confusion(cm, class_names, title, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main(config_path):
    config = OmegaConf.load(config_path)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dynamically construct output path using $DATA env var
    data_root = os.environ.get("DATA")
    if data_root is None:
        raise EnvironmentError("❌ The $DATA environment variable is not set.")

    config.output_dir = os.path.join(
        data_root,
        "AGENDA-Headset-Algorithm/outputs",
        #config.run_id or config.experiment_name
        "run_fold_1"
    )

    # Path to the dataset to be used (dataset name is specified in the yaml config file)
    config.data_path = os.path.join(
        data_root,
        "AGENDA-Headset-Algorithm/data/final_processed/",
        config.dataset_name+".h5"
    )

    label_map = {
        'neurotypical': 0,
        'epileptic': 1,
        'focal': 2,
        'generalized': 3,
        'left': 4,
        'right': 5
    }

    if config.k_folds > 1:
        _, _, test_dataset = create_kfold_stratified_datasets(
            h5_path=config.data_path,
            dataset_name=config.dataset_name,
            label_map=label_map,
            k_folds=config.k_folds,
            fold_index=config.fold_index,
            seed=config.seed
        )
    else:
        _, _, test_dataset = create_stratified_datasets(
            h5_path=config.data_path,
            dataset_name=config.dataset_name,
            label_map=label_map,
            ratios=(0.7, 0.15, 0.15),
            seed=config.seed
        )

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    #best_model_path = os.path.join(config.output_dir, "best_model.pt")
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
    class_names = [
        ["Neurotypical", "Epileptic"],
        ["Focal", "Generalized"],
        ["Left", "Right"]
    ]

    for i, level in enumerate(levels):
        valid = all_masks[:, i].bool()
        if valid.sum() == 0:
            continue

        y_true = all_targets[valid, i].cpu().numpy()
        y_pred = all_preds[valid, i].cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)

        output_file = os.path.join(config.output_dir, f"confusion_matrix_{level}.png")
        plot_confusion(cm, class_names[i], f"Confusion Matrix - {level.capitalize()}", output_file)
        print(f"✅ Saved {level} confusion matrix to {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    args = parser.parse_args()
    main(args.config)
