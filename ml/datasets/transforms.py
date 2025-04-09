import numpy as np
import random

class EEGNoiseTransform:
    """
    Adds Gaussian noise to EEG data to improve generalization.

    Parameters
    ----------
    std : float, optional (default=0.05)
        Standard deviation of the Gaussian noise to add.

    Example
    -------
    transform = EEGNoiseTransform(std=0.02)
    augmented = transform(eeg_data)
    """

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, x):
        return x + np.random.normal(0, self.std, x.shape)


class TimeShiftTransform:
    """
    Applies a random circular shift along the time axis of EEG data.

    Parameters
    ----------
    max_shift : int, optional (default=20)
        Maximum number of time steps to shift in either direction.

    Example
    -------
    transform = TimeShiftTransform(max_shift=10)
    shifted = transform(eeg_data)
    """

    def __init__(self, max_shift=20):
        self.max_shift = max_shift

    def __call__(self, x):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        return np.roll(x, shift, axis=-1)


class RandomEpochDropout:
    """
    Randomly zeroes out a subset of epochs to simulate missing segments.

    Parameters
    ----------
    drop_prob : float, optional (default=0.1)
        Proportion of epochs to zero out.

    Example
    -------
    transform = RandomEpochDropout(drop_prob=0.2)
    dropped = transform(eeg_data)
    """

    def __init__(self, drop_prob=0.1):
        self.drop_prob = drop_prob

    def __call__(self, x):
        x = x.copy()
        n_epochs = x.shape[0]
        n_drop = int(self.drop_prob * n_epochs)
        drop_idxs = np.random.choice(n_epochs, size=n_drop, replace=False)
        x[drop_idxs] = 0
        return x


class ChannelMaskTransform:
    """
    Randomly masks entire EEG channels across all epochs.

    Parameters
    ----------
    mask_prob : float, optional (default=0.1)
        Proportion of channels to mask.

    Example
    -------
    transform = ChannelMaskTransform(mask_prob=0.15)
    masked = transform(eeg_data)
    """

    def __init__(self, mask_prob=0.1):
        self.mask_prob = mask_prob

    def __call__(self, x):
        x = x.copy()
        n_channels = x.shape[1]
        n_mask = int(self.mask_prob * n_channels)
        mask_idxs = np.random.choice(n_channels, size=n_mask, replace=False)
        x[:, mask_idxs, :] = 0
        return x

class Compose:
    """
    Composes several transforms together and applies them sequentially.

    Parameters
    ----------
    transforms : list of callable
        List of transformation functions to apply.

    Example
    -------
    transform = Compose([
        EEGNoiseTransform(std=0.01),
        TimeShiftTransform(max_shift=10)
    ])
    output = transform(eeg_data)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x
