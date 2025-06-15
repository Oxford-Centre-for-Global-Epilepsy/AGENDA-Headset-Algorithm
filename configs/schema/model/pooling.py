from dataclasses import dataclass

@dataclass
class AttentionParams:
    hidden_dim: int = 64
    dropout_rate: float = 0.5
    kernel_length: int = 64
     
@dataclass
class AttentionPoolingConfig:
    _target_: str = "ml.models.pooling.attention.AttentionPooling"
    type: str = "attention"
    params: AttentionParams = AttentionParams()


@dataclass
class MeanPoolingConfig:
    _target_: str = "ml.models.pooling.mean.MeanPooling"
    type: str = "mean"
    params: None


@dataclass
class TransformerParams:
    num_heads: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.1
    use_cls_token: bool = True
     
@dataclass
class TransformerPoolingConfig:
    _target_: str = "ml.models.pooling.transformer.TransformerPooling"
    type: str = "transformer"
    params: TransformerParams = TransformerParams()
