import os
from pathlib import Path

import json
import numpy as np

import matplotlib.pyplot as plt
from math import ceil, sqrt
from matplotlib import colormaps

def summarize_fold_metrics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Print parameters
    params = data.get("params", {})
    print("Parameters:")
    print(json.dumps(params, indent=2))

    history = data["history"]

    f1_scores = []
    auc_scores = []
    test_f1_scores = []
    test_auc_scores = []

    summary_per_fold = {}

    for fold, epochs in history.items():
        # Extract validation metrics per epoch
        fold_f1s = [epoch.get("f1", 0.0) for epoch in epochs]
        fold_aucs = [epoch.get("auc", 0.0) for epoch in epochs]

        # Find best validation F1 and corresponding index
        best_f1 = max(fold_f1s)
        best_idx = fold_f1s.index(best_f1)
        best_auc = fold_aucs[best_idx] if fold_aucs else None

        # Get test metrics at best validation epoch
        test_f1 = epochs[best_idx].get("test_f1")
        test_auc = epochs[best_idx].get("test_auc")

        # Append to lists for aggregation
        f1_scores.append(best_f1)
        if best_auc is not None:
            auc_scores.append(best_auc)
        if test_f1 is not None:
            test_f1_scores.append(test_f1)
        if test_auc is not None:
            test_auc_scores.append(test_auc)

        # Store per-fold summary
        summary_per_fold[int(fold)] = {
            "best_F1": best_f1,
            # "best_AUC": best_auc,
            "test_F1": test_f1,
            # "test_AUC": test_auc,
        }

    # Aggregate summary
    summary = {
        "per_fold": summary_per_fold,
        "F1_mean": float(np.mean(f1_scores)),
        "F1_std": float(np.std(f1_scores)),
    }

    if test_f1_scores:
        summary["test_F1_mean"] = float(np.mean(test_f1_scores))
        summary["test_F1_std"] = float(np.std(test_f1_scores))

    return summary

def plot_fold_metrics_grid(json_paths, save_path=None):
    """
    Plot F1 trajectories for multiple result JSONs in a grid layout.
    Each subplot corresponds to one JSON file, with best epochs reported.
    """
    num_files = len(json_paths)
    cols = ceil(sqrt(num_files))
    rows = ceil(num_files / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Validation and Test F1 Trajectories", fontsize=16)

    for idx, json_path in enumerate(json_paths):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        with open(json_path, "r") as f:
            data = json.load(f)

        history = data["history"]
        summary_per_fold = {}

        # Assign unique color to each fold
        cmap = colormaps['tab10']        
        fold_ids = sorted(map(int, history.keys()))
        color_map = {fold: cmap(i % 10) for i, fold in enumerate(fold_ids)}

        for fold_str, epochs in history.items():
            fold = int(fold_str)
            val_f1s = [ep.get("f1", 0.0) for ep in epochs]
            test_f1s = [ep.get("test_f1", np.nan) for ep in epochs]

            # Find best epoch (1-based index)
            best_idx = int(np.argmax(val_f1s))
            best_epoch = best_idx + 1  
            best_val_f1 = val_f1s[best_idx]
            best_test_f1 = test_f1s[best_idx] if best_idx < len(test_f1s) else None

            summary_per_fold[fold] = {
                "best_epoch": best_epoch,
                "best_F1": best_val_f1,
                "test_F1": best_test_f1,
            }

            # Plot validation and test F1 using same color, different linestyle
            ax.plot(val_f1s, label=f"Fold {fold} - Val", color=color_map[fold], linestyle='-')
            ax.plot(test_f1s, label=f"Fold {fold} - Test", color=color_map[fold], linestyle='--')

            # Mark best epoch with a dot
            ax.scatter(best_epoch-1, best_val_f1, color=color_map[fold], marker='o', edgecolor='k', zorder=5)
            ax.text(best_epoch-1, best_val_f1 + 0.01, f"E{best_epoch}", fontsize=7, ha='center')

        # Use 'name' field from params if available
        title = data.get("params", {}).get("name", json_path.stem)
        ax.set_title(title)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1 Score")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_ylim(0.3, 0.9)

        # Optional: also print summary in console
        print(f"Results for {title}:")
        for fold, res in summary_per_fold.items():
            print(f"  Fold {fold}: Best Epoch={res['best_epoch']}, "
                  f"Val F1={res['best_F1']:.3f}, Test F1={res['test_F1']:.3f}")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    parent_dir = Path(__file__).resolve().parent.parent
    target_dir = parent_dir / "training" / "results"  # <-- Path object now

    json_files = sorted(target_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON result files found in training/results/")

    # Plot all in one grid
    plot_fold_metrics_grid(json_files, save_path=target_dir / "f1_trajectories_grid.png")

    for json_file in json_files:
        print(summarize_fold_metrics(json_file))
        print("="*50)