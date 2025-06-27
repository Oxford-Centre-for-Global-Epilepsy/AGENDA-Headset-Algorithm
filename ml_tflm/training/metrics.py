from ml_tflm.training.cast_prediction import cast_prediction_flat, cast_prediction_hierarchical, cast_labels
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

class metric_evaluator():
    def __init__(self, label_config, prediction_caster):
        """
        Initializes the metric evaluator with a label configuration.

        Args:
            label_config (dict): Dictionary containing label mappings and prior probabilities.
        """

        # Load label map and prior probabilities from the configuration
        self.label_map = label_config["label_map"]
        self.label_map_internal = {key: i for i, key in enumerate(self.label_map.keys())}

        flat_index_map = {
            "neurotypical": 0,
            "generalized": 1,
            "left": 2,
            "right": 3
        }

        self.flat_index_map_internal = {self.label_map_internal[k]: v for k, v in flat_index_map.items()}

        # Load the caster function for predictions
        self.prediction_caster = prediction_caster

    def evaluate(self, preds, targets, ignore_index=-1):
        """
        Evaluates the predictions against the targets and computes metrics.

        Args:
            preds: list of dicts (model outputs for each batch)
            targets: list of str or int ground-truth labels (flattened)
            ignore_index: int, value in preds/targets to ignore (default: -1)

        Returns:
            dict with macro F1, accuracy, precision, recall, and confusion matrix
        """
        # Cast predictions using the caster (handles list of dicts internally)
        pred_indices = np.array(self.prediction_caster(preds))  # shape [B]

        # Convert true labels to indices
        target_indices = np.array(cast_labels(targets, self.flat_index_map_internal))

        # Mask out invalid entries
        valid_mask = (target_indices != ignore_index) & (pred_indices != ignore_index)
        pred_indices = pred_indices[valid_mask]
        target_indices = target_indices[valid_mask]

        if len(pred_indices) == 0:
            return {
                "f1": None,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "confusion_matrix": None
            }

        return {
            "f1": f1_score(target_indices, pred_indices, average="macro"),
            "accuracy": accuracy_score(target_indices, pred_indices),
            "precision": precision_score(target_indices, pred_indices, average="macro"),
            "recall": recall_score(target_indices, pred_indices, average="macro"),
            "confusion_matrix": confusion_matrix(target_indices, pred_indices)
        }
