import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanPooling(nn.Module):
    """Simple mean pooling over epochs, with optional masking."""
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None, return_weights=False):
        # x: [B, E, D]
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1)  # [B, E, 1]
            masked = x * mask
            pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        if return_weights:
            # Uniform weights
            weights = torch.ones(x.size(0), x.size(1), device=x.device)
            if mask is not None:
                weights = weights * mask
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            else:
                weights = weights / x.size(1)
            return pooled, weights

        return pooled


class AttentionPooling(nn.Module):
    """
    Learnable attention pooling over epochs.

    Parameters
    ----------
    input_dim : int
        Feature dimension of each epoch after EEGNet.
    hidden_dim : int
        Hidden layer size for computing attention scores.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None, return_weights=False):
        # x: [B, E, D]
        scores = self.attn_net(x).squeeze(-1)  # [B, E]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=1)  # [B, E]
        weighted_sum = torch.sum(x * weights.unsqueeze(-1), dim=1)  # [B, D]

        if return_weights:
            return weighted_sum, weights  # [B, D], [B, E]

        return weighted_sum


class TransformerPooling(nn.Module):
    """
    Transformer-based pooling over epochs using a TransformerEncoder.

    Parameters
    ----------
    input_dim : int
        Dimension of EEGNet features (D).
    num_heads : int
        Number of self-attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout in transformer.
    use_cls_token : bool
        If True, uses a [CLS] token. Else, uses mean pooling over outputs.
    """
    def __init__(self, input_dim, num_heads=4, num_layers=2, dropout=0.1, use_cls_token=True):
        super().__init__()
        self.use_cls_token = use_cls_token
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))  # [1, 1, D]

    def forward(self, x, mask=None, return_weights=False):
        # x: [B, E, D]
        B, E, D = x.shape

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, 1, D)  # [B, 1, D]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, E+1, D]

            if mask is not None:
                cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)  # [B, E+1]

        src_key_padding_mask = ~mask if mask is not None else None
        encoded = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, E(+1), D]

        if self.use_cls_token:
            pooled = encoded[:, 0]  # [B, D]
            if return_weights:
                return pooled, None  # CLS pooling doesn't expose weights directly
            return pooled
        else:
            pooled = encoded.mean(dim=1)  # [B, D]
            if return_weights:
                # Uniform weights (optional, for consistency)
                weights = torch.ones(B, E, device=x.device)
                if mask is not None:
                    weights = weights * mask
                    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
                else:
                    weights = weights / E
                return pooled, weights

            return pooled
