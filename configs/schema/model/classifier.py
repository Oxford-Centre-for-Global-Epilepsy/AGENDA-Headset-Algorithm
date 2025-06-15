from dataclasses import dataclass, field
from typing import List

@dataclass
class MultiHeadParams:
    num_classes: List[int] = field(default_factory=lambda: [2, 2, 2])  # Number of classes for each level
    hidden_dim: int = 128   # Hidden dimension for classifiers
    
@dataclass
class MultiHeadClassifierConfig:
    _target_: str = "ml.models.classifier.multihead.MultiHeadClassifier"
    type: str = "multihead"
    params: MultiHeadParams = MultiHeadParams()
