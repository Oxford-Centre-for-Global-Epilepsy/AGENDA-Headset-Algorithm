from ml_tflm.training.cast_prediction import cast_labels, CASTER_REGISTRY
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

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

    def evaluate(self, preds, targets, ignore_index=-1):
        """
        Evaluates the predictions against the targets and computes both flat and hierarchical metrics.

        Args:
            preds: list of dicts (model outputs for each batch)
            targets: list of str or int ground-truth labels (flattened)
            ignore_index: int, value in preds/targets to ignore (default: -1)

        Returns:
            dict with macro F1, accuracy, precision, recall, confusion matrix, and hierarchical metrics
        """
        # ---- Flat metrics ----
        pred_indices = np.array(self.prediction_caster(preds))  # shape [B]
        target_indices = np.array(cast_labels(targets, self.flat_index_map_internal))

        valid_mask = (target_indices != ignore_index) & (pred_indices != ignore_index)
        pred_indices_valid = pred_indices[valid_mask]
        target_indices_valid = target_indices[valid_mask]

        results = {
            "f1": f1_score(target_indices_valid, pred_indices_valid, average="macro", zero_division=0),
            "accuracy": accuracy_score(target_indices_valid, pred_indices_valid),
            "precision": precision_score(target_indices_valid, pred_indices_valid, average="macro", zero_division=0),
            "recall": recall_score(target_indices_valid, pred_indices_valid, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(target_indices_valid, pred_indices_valid)
        }

        if len(pred_indices_valid) == 0:
            return results

        # ---- Hierarchical metrics ----
        try:
            pred_l1, pred_l2, pred_l3 = map(np.array, self.prediction_caster(preds, return_hierarchical=True))
        except Exception:
            print("Hierarchical prediction not supported by caster")
            return results  # fallback if caster doesn't support hierarchical

        # Map ground-truth flat label to hierarchical levels
        # flat labels: 0=neurotypical, 1=generalized, 2=left, 3=right
        t_l1 = np.where(target_indices == 0, 0, 1)  # neurotypical vs epileptic
        t_l2 = np.where(target_indices == 1, 1, 0)  # generalized vs focal
        t_l3 = np.where(target_indices == 3, 1, 0)  # right vs left

        hierarchical_metrics = {}

        def compute_level_metrics(y_true, y_pred, mask, prefix):
            if np.sum(mask) == 0:
                print(f"{prefix.upper()} - No valid samples.")
                return {
                    f"{prefix}_f1": None,
                    f"{prefix}_acc": None,
                    f"{prefix}_precision": None,
                    f"{prefix}_recall": None
                }
            f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            acc = accuracy_score(y_true[mask], y_pred[mask])
            prec = precision_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            rec = recall_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
            return {
                f"{prefix}_f1": f1,
                f"{prefix}_acc": acc,
                f"{prefix}_precision": prec,
                f"{prefix}_recall": rec,
            }

        mask_l1 = target_indices != ignore_index
        hierarchical_metrics.update(compute_level_metrics(t_l1, pred_l1, mask_l1, "level1"))

        mask_l2 = (t_l1 == 1) & (target_indices != ignore_index)
        hierarchical_metrics.update(compute_level_metrics(t_l2, pred_l2, mask_l2, "level2"))

        mask_l3 = (t_l1 == 1) & (t_l2 == 0) & (target_indices != ignore_index)
        hierarchical_metrics.update(compute_level_metrics(t_l3, pred_l3, mask_l3, "level3"))

        results.update(hierarchical_metrics)
        return results
