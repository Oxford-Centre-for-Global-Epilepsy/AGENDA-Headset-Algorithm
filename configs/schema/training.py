from dataclasses import dataclass
from typing import List, Optional

@dataclass
class HierarchicalLossParams:
    weights: List[float] = (1.0, 1.0, 1.0)
    level1_weights: Optional[List[float]] = None
    level2_weights: Optional[List[float]] = None
    level3_weights: Optional[List[float]] = None

@dataclass
class LossConfig:
    type: str = "hierarchical_loss"
    params: HierarchicalLossParams = HierarchicalLossParams()

@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 0.001

@dataclass
class TrainingLoopConfig:
    batch_size: int = 4
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: str = "Adam"
    resume: bool = False
    checkpoint_path: Optional[str] = None

@dataclass
class TrainingConfig:
    losses: LossConfig = LossConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    training: TrainingLoopConfig = TrainingLoopConfig()
