import numpy as np
import torch

def compute_vanilla_saliency(model, input_tensor, class_idx=None):
    """
    Computes vanilla saliency map: gradient of output class score w.r.t. input.

    Args:
        model (nn.Module): Your EEGNet-based model or feature extractor.
        input_tensor (Tensor): Shape [1, 1, C, T] - single epoch input.
        class_idx (int, optional): Target class index. If None, uses predicted class.

    Returns:
        saliency_map (Tensor): Gradient magnitude, same shape as input [1, 1, C, T]
    """
    model.eval()
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(input_tensor)  # [1, num_classes] or feature vector
    if class_idx is None:
        class_idx = output.argmax(dim=1).item()
    score = output[:, class_idx]

    # Backward
    model.zero_grad()
    score.backward()

    saliency = input_tensor.grad.data.abs()
    return saliency  # [1, 1, C, T]

def simplify_saliency_map(saliency_tensor, reduce="none", normalize=True):
    """
    Simplifies and optionally normalizes a raw saliency map for visualization.

    Args:
        saliency_tensor (Tensor): [1, 1, C, T]
        reduce (str): One of 'none', 'channel', 'time'
            - 'none' → [C, T]
            - 'channel' → [T] (mean/max over channels)
            - 'time' → [C] (mean/max over time)
        normalize (bool): Whether to min-max normalize output

    Returns:
        np.ndarray: simplified saliency map
    """
    sal = saliency_tensor.squeeze().detach().cpu().numpy()  # [C, T]

    if reduce == "channel":
        sal = sal.mean(axis=0)
    elif reduce == "time":
        sal = sal.mean(axis=1)

    if normalize:
        sal -= sal.min()
        sal /= sal.max() + 1e-8

    return sal