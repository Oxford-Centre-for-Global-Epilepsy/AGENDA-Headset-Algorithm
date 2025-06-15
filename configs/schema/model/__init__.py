from dataclasses import dataclass
from omegaconf import DictConfig  # because these are dynamically loaded

@dataclass
class ModelConfig:
    feature_extractor: DictConfig
    pooling: DictConfig
    classifier: DictConfig
    