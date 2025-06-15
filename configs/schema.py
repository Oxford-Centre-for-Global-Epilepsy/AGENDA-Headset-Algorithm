# ml/configs/schema.py
from dataclasses import dataclass
from typing import Union
from omegaconf import MISSING

@dataclass
class MultiHeadParams:
    num_classes: List[int] = field(default_factory=lambda: [2, 2, 2])  # Number of classes for each level
    hidden_dim: int = 128   # Hidden dimension for classifiers
    
@dataclass
class MultiHeadClassifierConfig:
    _target_: str = "ml.models.classifier.multihead.MultiHeadClassifier"
    type: str = "multihead"
    params: MultiHeadParams = MultiHeadParams()

@dataclass
class DatasetSiteConfig:
    project_name: str = "AGENDA"
    site_name: str = "South Africa"
    dataset_name: str = "EEG"
    dataset_path: str = "${env:DATA}/AGENDA-Headset-Algorithm/data/..."

