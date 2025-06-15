import hydra
from omegaconf import OmegaConf, DictConfig
import sys
import os
import csv
import ast
import torch
import uuid
import random
import numpy as np
from sklearn.utils import class_weight
from datetime import datetime
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from ml.datasets.eeg_dataset import EEGRecordingDataset
from ml.training.metrics import compute_per_level_metrics
from ml.training.losses.hierarchical_loss import HierarchicalLoss
from ml.training.utils import EarlyStopping
from ml.utils.splits import load_split_indices
from ml.models.hierarchical_classifier import HierarchicalClassifier

from configs.schema import MainConfig
from hydra.utils import instantiate

# don't buffer the standard output, i.e. write straight to the file when able
sys.stdout.reconfigure(line_buffering=True)

def generate_run_name(config):
    parts = [f"fold_{config.dataset.fold_index}"]

    drop_electrodes = config.dataset.get("drop_electrodes")
    if isinstance(drop_electrodes, str):
        try:
            drop_electrodes = ast.literal_eval(drop_electrodes)
        except (ValueError, SyntaxError):
            drop_electrodes = [ch.strip() for ch in drop_electrodes.split(',')]

    if drop_electrodes:
        omitted = "_".join(sorted(drop_electrodes))
        parts.append(f"omit_{omitted}")

    return "_".join(parts)

def generate_tags(config):
    tags = {
        "experiment_name": config.experiment_name,
        "fold": str(config.dataset.fold_index),
    }
    if config.dataset.get("drop_electrodes"):
        tags["drop_electrodes"] = ",".join(sorted(config.dataset.drop_electrodes))
    return tags

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"🔒 Set global random seed to: {seed}")

def save_checkpoint(path, model, optimizer, epoch, best_val_loss):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss
    }, path)

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_val_loss"]

def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_all_bytes = param_size + buffer_size
    return size_all_bytes / 1e6  # Convert to MB

@hydra.main(config_path="../../configs", config_name="config", version_base="1.1")
def main(cfg: MainConfig):
    
    # Hydra will resolve ${env:...} automatically when accessed
    print(f"Running experiment with configuration:\n{OmegaConf.to_yaml(cfg)}")

    dataset_cfg = cfg.dataset
    print(f"Project Name: {dataset_cfg.project_name}")
    print(f"Site Name: {dataset_cfg.site_name}")
    print(f"Dataset Name: {dataset_cfg.dataset_name}")
    print(f"Dataset Path: {dataset_cfg.dataset_path}")  # This resolves ${env:DATA} automatically

    model_cfg = cfg.model
    print("Model configuration:")
    print(f"  Feature Extractor: {model_cfg.feature_extractor._target_}")
    print(f"  Pooling Layer:     {model_cfg.pooling._target_}")
    print(f"  Classifier:        {model_cfg.classifier._target_}")

if __name__ == "__main__":
    main()
