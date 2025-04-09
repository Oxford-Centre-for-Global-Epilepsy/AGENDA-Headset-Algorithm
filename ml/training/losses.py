import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_cross_entropy(logits, targets, mask):
    """
    Computes cross-entropy loss while ignoring entries where mask == 0.
    
    Args:
        logits: [B, C]
        targets: [B]
        mask: [B] - bool tensor
    
    Returns:
        Scalar loss
    """
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits = logits[mask]
    targets = targets[mask]
    return F.cross_entropy(logits, targets)


class HierarchicalLoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.weights = weights

    def forward(self, outputs, targets, label_mask):
        """
        Args:
            outputs: dict with keys 'level1_logits', 'level2_logits', 'level3_logits'
            targets: tensor [B, 3]
            label_mask: tensor [B, 3] - 1 = valid label
        
        Returns:
            Total masked hierarchical loss
        """
        loss1 = masked_cross_entropy(outputs["level1_logits"], targets[:, 0], label_mask[:, 0])
        loss2 = masked_cross_entropy(outputs["level2_logits"], targets[:, 1], label_mask[:, 1])
        loss3 = masked_cross_entropy(outputs["level3_logits"], targets[:, 2], label_mask[:, 2])

        return self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss3
