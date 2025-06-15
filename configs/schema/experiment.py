from dataclasses import dataclass, field
from typing import List

@dataclass
class AblationExperimentConfig:
    name: str = "electrode_ablation"
    output_dir: str = "results/electrode_ablation"
    omit_channels: List[str] = field(default_factory=lambda: ["A1", "A2"])

