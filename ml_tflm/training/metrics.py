from ml_tflm.training.cast_prediction import cast_labels, CASTER_REGISTRY
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import re

class metric_evaluator():
    def __init__(self, label_config, prediction_caster: str):
        """
        Initializes the metric evaluator with a label configuration.

        Args:
            label_config (dict): Dictionary containing label mappings and prior probabilities.
        """

        # Load label map and prior probabilities from the configuration
        self.label_map = label_config["label_map"]
        self.label_map_internal = {key: i for i, key in enumerate(self.label_map.keys())}
        self.label_prior = label_config["label_prior"]

        flat_index_map = {
            "neurotypical": 0,
            "generalized": 1,
            "left": 2,
            "right": 3
        }

        self.flat_index_map_internal = {self.label_map_internal[k]: v for k, v in flat_index_map.items()}

        # Load the caster function for predictions
        self.prediction_caster = CASTER_REGISTRY[prediction_caster]
        self.binary_flag = bool(re.search(r"binary$", prediction_caster))

    def evaluate(self, preds, targets, ignore_index=-1):
        if self.binary_flag:
            return self.binary_eval(preds, targets, ignore_index)
        else:
            return self.quad_eval(preds, targets, ignore_index)

    def binary_eval(self, preds, targets, ignore_index):
        pred_indices = np.array(self.prediction_caster(preds))
        target_indices = np.array([
                                    0 if self.flat_index_map_internal[y] == 0 else 1
                                    for y in targets.numpy()
                                ])


        valid_mask = (target_indices != ignore_index) & (pred_indices != ignore_index)
        pred_valid = pred_indices[valid_mask]
        target_valid = target_indices[valid_mask]

        return {
            "f1": f1_score(target_valid, pred_valid, average="binary", zero_division=0),
            "accuracy": accuracy_score(target_valid, pred_valid),
            "precision": precision_score(target_valid, pred_valid, average="binary", zero_division=0),
            "recall": recall_score(target_valid, pred_valid, average="binary", zero_division=0),
            "confusion_matrix": confusion_matrix(target_valid, pred_valid)
        }

    def quad_eval(self, preds, targets, ignore_index):
        pred_indices = np.array(self.prediction_caster(preds))
        target_indices = np.array(cast_labels(targets, self.flat_index_map_internal))

        valid_mask = (target_indices != ignore_index) & (pred_indices != ignore_index)
        pred_valid = pred_indices[valid_mask]
        target_valid = target_indices[valid_mask]

        results = {
            "f1": f1_score(target_valid, pred_valid, average="macro", zero_division=0),
            "accuracy": accuracy_score(target_valid, pred_valid),
            "precision": precision_score(target_valid, pred_valid, average="macro", zero_division=0),
            "recall": recall_score(target_valid, pred_valid, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(target_valid, pred_valid)
        }

        if len(pred_valid) == 0:
            return results

        # ---- Hierarchical metrics ----
        try:
            pred_l1, pred_l2, pred_l3 = map(np.array, self.prediction_caster(preds, return_hierarchical=True))
        except Exception:
            print("Hierarchical prediction not supported by caster")
            results["hierarchical_skipped"] = True
            return results

        t_l1 = np.where(target_indices == 0, 0, 1)  # neurotypical vs epileptic
        t_l2 = np.where(target_indices == 1, 1, 0)  # generalized vs focal
        t_l3 = np.where(target_indices == 3, 1, 0)  # right vs left

        def compute_level_metrics(y_true, y_pred, mask, prefix):
            if np.sum(mask) == 0:
                print(f"{prefix.upper()} - No valid samples.")
                return {
                    f"{prefix}_f1": None,
                    f"{prefix}_acc": None,
                    f"{prefix}_precision": None,
                    f"{prefix}_recall": None
                }
            return {
                f"{prefix}_f1": f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0),
                f"{prefix}_acc": accuracy_score(y_true[mask], y_pred[mask]),
                f"{prefix}_precision": precision_score(y_true[mask], y_pred[mask], average="macro", zero_division=0),
                f"{prefix}_recall": recall_score(y_true[mask], y_pred[mask], average="macro", zero_division=0),
            }

        hierarchical_metrics = {}
        mask_l1 = target_indices != ignore_index
        hierarchical_metrics.update(compute_level_metrics(t_l1, pred_l1, mask_l1, "level1"))

        mask_l2 = (t_l1 == 1) & (target_indices != ignore_index)
        hierarchical_metrics.update(compute_level_metrics(t_l2, pred_l2, mask_l2, "level2"))

        mask_l3 = (t_l1 == 1) & (t_l2 == 0) & (target_indices != ignore_index)
        hierarchical_metrics.update(compute_level_metrics(t_l3, pred_l3, mask_l3, "level3"))

        results.update(hierarchical_metrics)
        return results