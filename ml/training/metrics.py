from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
import torch

def compute_per_level_metrics(preds, targets, masks, label_names=None):
    """
    Computes accuracy and F1 for each hierarchy level.

    Args:
        preds: Tensor [B, 3] - predicted class indices at each level
        targets: Tensor [B, 3] - true class indices (global labels)
        masks: Tensor [B, 3] - label validity mask (bool or 0/1)
        label_names: Optional dict of {level_index: {int: str}} for pretty printing

    Returns:
        metrics: dict {level: {'accuracy': float, 'f1': float}}
    """
    metrics = {}

    for level in range(3):
        valid = masks[:, level].bool()
        if valid.sum() == 0:
            metrics[f"level{level+1}"] = {"accuracy": None, "f1": None, "precision": None, "recall": None, "roc_auc": None}
            continue

        y_true = targets[valid, level]
        y_pred = preds[valid, level]

        # Apply label remapping to ensure 0/1 targets for levels 2 and 3
        if level == 1:  # Level 2: focal (2) → 0, generalized (3) → 1
            y_true = torch.where(y_true == 3, 1, 0)
        elif level == 2:  # Level 3: left (4) → 0, right (5) → 1
            y_true = torch.where(y_true == 5, 1, 0)

        acc = accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
        precision = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
        recall = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")
        roc_auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average="macro")

        metrics[f"level{level+1}"] = {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc
        }

    return metrics