#!/usr/bin/env python3

import os
import json
import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

from ml.datasets.eeg_dataset import EEGRecordingDataset


def compute_label_distributions(subject_ids, subject_to_labels):
    level1 = Counter()
    level2 = Counter()
    level3 = Counter()

    for sid in subject_ids:
        labels = subject_to_labels[sid]
        if len(labels) > 0:
            level1[labels[0]] += 1
        if len(labels) > 1:
            level2[labels[1]] += 1
        if len(labels) > 2:
            level3[labels[2]] += 1

    return {
        "level1": dict(level1),
        "level2": dict(level2),
        "level3": dict(level3)
    }


def plot_distribution(dist, title, save_path):
    sns.set(style="whitegrid")
    levels = ['level1', 'level2', 'level3']
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))

    for i, level in enumerate(levels):
        labels = list(dist[level].keys())
        counts = list(dist[level].values())
        sns.barplot(x=labels, y=counts, ax=axs[i], hue=labels, palette="muted", legend=False)
        axs[i].set_title(f"{level.capitalize()} Classes")
        axs[i].set_ylabel("Count")
        axs[i].set_xticks(range(len(labels)))
        axs[i].set_xticklabels(labels, rotation=45)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()


def build_multilabel_matrix(subject_ids, subject_to_labels):
    matrix = []
    for sid in subject_ids:
        labels = subject_to_labels[sid]
        l1 = labels[0] if len(labels) > 0 else None
        l2 = labels[1] if len(labels) > 1 else None
        l3 = labels[2] if len(labels) > 2 else None

        row = [
            int(l1 == "neurotypical"),
            int(l1 == "epileptic"),
            int(l2 == "generalized"),
            int(l2 == "focal"),
            int(l3 == "left"),
            int(l3 == "right"),
            int(l1 == "epileptic" and l2 == "focal" and l3 == "left"),
            int(l1 == "epileptic" and l2 == "focal" and l3 == "right"),
            int(l1 == "epileptic" and l2 == "generalized"),
        ]
        matrix.append(row)

    return np.array(matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--output_dir", required=False, help="If not provided, defaults to folder alongside HDF5 file.")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Check for the output directory
    if args.output_dir is None:
        h5_base_dir = os.path.dirname(os.path.abspath(args.h5_path))
        args.output_dir = os.path.join(h5_base_dir, f"dataset_splits")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"📂 Loading subject labels from {args.h5_path}")
    dataset = EEGRecordingDataset(args.h5_path, args.dataset_name)
    subject_to_labels = dataset.get_subjects_with_labels()
    subject_ids = list(subject_to_labels.keys())
    label_matrix = build_multilabel_matrix(subject_ids, subject_to_labels)

    # Perform stratified test split
    print("🔀 Performing stratified train+val / test split using multilabel stratification")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    trainval_idx, test_idx = next(msss.split(subject_ids, label_matrix))
    trainval_ids = [subject_ids[i] for i in trainval_idx]
    test_ids = [subject_ids[i] for i in test_idx]

    # Save test split
    with open(os.path.join(args.output_dir, "test_split.json"), "w") as f:
        json.dump({"test": test_ids}, f, indent=2)

    test_dist = compute_label_distributions(test_ids, subject_to_labels)
    plot_distribution(test_dist, "Test Set Distribution", os.path.join(args.output_dir, "test_distribution.png"))
    print(f"📊 Test Distribution:\n  L1: {test_dist['level1']}\n  L2: {test_dist['level2']}\n  L3: {test_dist['level3']}")

    summary = {"test": {"test": test_dist}}

    print(f"🔄 Creating {args.k_folds}-fold stratified CV on training+val data ({len(trainval_ids)} subjects)")
    kfold_matrix = build_multilabel_matrix(trainval_ids, subject_to_labels)
    skf = MultilabelStratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    for i, (train_idx, val_idx) in enumerate(skf.split(trainval_ids, kfold_matrix)):
        train_ids = [trainval_ids[j] for j in train_idx]
        val_ids = [trainval_ids[j] for j in val_idx]
        fold_name = f"fold_{i}"

        # Save fold split
        with open(os.path.join(args.output_dir, f"{fold_name}.json"), "w") as f:
            json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

        # Compute and store distributions
        train_dist = compute_label_distributions(train_ids, subject_to_labels)
        val_dist = compute_label_distributions(val_ids, subject_to_labels)

        summary[fold_name] = {"train": train_dist, "val": val_dist}

        # Plot
        plot_distribution(train_dist, f"{fold_name} - Train", os.path.join(args.output_dir, f"{fold_name}_train_distribution.png"))
        plot_distribution(val_dist, f"{fold_name} - Val", os.path.join(args.output_dir, f"{fold_name}_val_distribution.png"))

        print(f"✅ {fold_name}: train={len(train_ids)}, val={len(val_ids)}")
        print(f"📊 {fold_name} — Train:\n  L1: {train_dist['level1']}\n  L2: {train_dist['level2']}\n  L3: {train_dist['level3']}")
        print(f"📊 {fold_name} — Val:\n  L1: {val_dist['level1']}\n  L2: {val_dist['level2']}\n  L3: {val_dist['level3']}")

    # Save final summary JSON
    summary_path = os.path.join(args.output_dir, "distribution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📁 Saved distribution summary JSON to: {summary_path}")
    print("🏁 Done generating dataset splits with enhanced multilabel stratification.")


if __name__ == "__main__":
    main()
