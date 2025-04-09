import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet for per-epoch EEG feature extraction.

    Parameters
    ----------
    num_channels : int
        Number of EEG channels (e.g., 19 for standard 10–20 montage).
    num_samples : int
        Number of time samples per epoch (e.g., 256 for 1-second @ 256Hz).
    dropout_rate : float
        Dropout rate applied after convolution layers (default = 0.5).
    F1 : int
        Number of temporal filters in the first conv layer.
    D : int
        Depth multiplier – number of spatial filters per temporal filter.
    F2 : int
        Number of separable (pointwise) filters after spatial filtering.
    kernel_length : int
        Size of temporal kernel in first conv layer (default = 64).
    """

    def __init__(self, num_channels=19, num_samples=256,
                 dropout_rate=0.5, F1=8, D=2, F2=16, kernel_length=64):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate)
        )

        self.feature_dim = self._compute_feature_dim(num_channels, num_samples)

    def _compute_feature_dim(self, C, T):
        x = torch.zeros(1, 1, C, T)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        return x.view(1, -1).shape[1]

    def forward(self, x, return_features=False):
        """
        Forward pass of EEGNet.

        Parameters
        ----------
        x : Tensor
            Shape: [batch, 1, channels, time]
        return_features : bool
            If True, returns a dict of intermediate features.

        Returns
        -------
        Tensor or dict
            If return_features is False: returns final feature vector [B, feature_dim].
            If return_features is True: returns a dict of intermediate activations.
        """
        x1 = self.firstconv(x)        # After temporal conv
        x2 = self.depthwiseConv(x1)   # After spatial filters
        x3 = self.separableConv(x2)   # Final conv output
        x_flat = x3.view(x3.size(0), -1)

        if return_features:
            return {
                "features": x_flat,             # Final feature vector
                "temporal_features": x1,        # After firstconv
                "spatial_features": x2,         # After depthwiseConv
                "final_features": x3            # Before flattening
            }

        return x_flat