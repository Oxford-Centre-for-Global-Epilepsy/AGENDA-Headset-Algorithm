from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_per_level_metrics(preds, targets, masks, label_names=None):
    """
    Computes accuracy and F1 for each hierarchy level.

    Args:
        preds: Tensor [B, 3] - predicted class indices at each level
        targets: Tensor [B, 3] - true class indices
        masks: Tensor [B, 3] - label validity mask (bool or 0/1)
        label_names: Optional dict of {level_index: {int: str}} for pretty printing

    Returns:
        metrics: dict {level: {'accuracy': float, 'f1': float}}
    """
    metrics = {}
    for level in range(3):
        valid = masks[:, level].bool()
        if valid.sum() == 0:
            metrics[f"level{level+1}"] = {"accuracy": None, "f1": None}
            continue

        y_true = targets[valid, level].cpu().numpy()
        y_pred = preds[valid, level].cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")

        metrics[f"level{level+1}"] = {
            "accuracy": acc,
            "f1": f1
        }
    return metrics
