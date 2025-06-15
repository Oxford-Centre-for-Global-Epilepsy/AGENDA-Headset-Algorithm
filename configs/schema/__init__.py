from dataclasses import dataclass
from configs.schema.dataset import DatasetConfig
from configs.schema.training import TrainingConfig
from configs.schema.experiment import AblationExperimentConfig
from configs.schema.model import ModelConfig

@dataclass
class MainConfig:
    experiment: AblationExperimentConfig
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig