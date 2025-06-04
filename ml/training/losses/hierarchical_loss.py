import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_cross_entropy(logits, targets, mask, class_weights=None):
    """
    Computes cross-entropy loss while ignoring entries where mask == 0.
    
    Args:
        logits: [B, C]
        targets: [B]
        mask: [B] - bool tensor
        class_weights: [C] - tensor of class weights, optional
    
    Returns:
        Scalar loss
    """

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    logits = logits[mask]
    targets = targets[mask]

    # If class_weights are provided, apply them to the cross-entropy loss
    if class_weights is not None:
        return F.cross_entropy(logits, targets, weight=class_weights)
    
    return F.cross_entropy(logits, targets)

class HierarchicalLoss(nn.Module):
    def __init__(self, weights=(1.0, 1.0, 1.0), level1_weights=None, level2_weights=None, level3_weights=None):
        super().__init__()
        self.weights = weights
        self.level1_weights = level1_weights  # Class weights for level 1
        self.level2_weights = level2_weights  # Class weights for level 2
        self.level3_weights = level3_weights  # Class weights for level 3

    def forward(self, outputs, targets, label_mask):
        """
        Args:
            outputs: dict with keys 'level1_logits', 'level2_logits', 'level3_logits'
            targets: tensor [B, 3] (global label indices)
            label_mask: tensor [B, 3] - 1 = valid label
        """

        # Level 1: 0 = neurotypical, 1 = epileptic (already correct)
        loss1 = masked_cross_entropy(outputs["level1_logits"], targets[:, 0], label_mask[:, 0], class_weights=self.level1_weights)

        # Level 2: remap 2=focal → 0, 3=generalized → 1
        level2_target = torch.where(targets[:, 1] == 3, 1, 0)
        loss2 = masked_cross_entropy(outputs["level2_logits"], level2_target, label_mask[:, 1], class_weights=self.level2_weights)

        # Level 3: remap 4=left → 0, 5=right → 1
        level3_target = torch.where(targets[:, 2] == 5, 1, 0)
        loss3 = masked_cross_entropy(outputs["level3_logits"], level3_target, label_mask[:, 2], class_weights=self.level3_weights)

        return self.weights[0] * loss1 + self.weights[1] * loss2 + self.weights[2] * loss3
