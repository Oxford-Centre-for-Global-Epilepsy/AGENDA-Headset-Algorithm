from dataclasses import dataclass

@dataclass
class EEGNetParams:
    num_channels: int = 21   
    num_samples: int = 129
    dropout_rate: float = 0.5
    F1: int = 8
    D: int = 2
    F2: int = 16
    kernel_length: int = 64
     
@dataclass
class EEGNetFeatureExtractorConfig:
    _target_: str = "ml.models.feature_extractor.eegnet.EEGNet"
    type: str = "eegnet"
    params: EEGNetParams = EEGNetParams()
