#!/usr/bin/env python3
"""
scripts/normalise_epoched_data.py

Usage:
    python scripts/normalise_epoched_data.py <input_fif> <output_fif>
"""

import sys
import os
import mne
import numpy as np

def normalise_epochs(epochs):
    """
    Normalise each channel independently for each epoch.
    
    Parameters:
    epochs : mne.Epochs
        The MNE Epochs object containing epoched EEG data.

    Returns:
    epochs : mne.Epochs
        The normalised MNE Epochs object.
    """
    data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

    # Compute mean and std per channel for each epoch
    mean_per_epoch = np.mean(data, axis=2, keepdims=True)  # Mean across time axis
    std_per_epoch = np.std(data, axis=2, keepdims=True)  # Std across time axis

    # Avoid division by zero (replace std=0 with 1 to keep data unchanged)
    std_per_epoch[std_per_epoch == 0] = 1.0

    # Normalise: (X - mean) / std [i.e. Z-score the epoched data channel independently]
    normalised_data = (data - mean_per_epoch) / std_per_epoch

    # Store the normalized data back in the MNE Epochs object
    epochs._data = normalised_data

    return epochs

    # Ensure output file has updated modification time
    try:
        os.utime(output_file, None)
        print(f"DEBUG: Touched output file: {output_file}", flush=True)
    except Exception as e:
        print(f"WARNING: Failed to update mtime: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/normalize_epoched_data.py <input_fif> <output_fif>")
        sys.exit(1)

    input_fif = sys.argv[1]
    output_fif = sys.argv[2]

    if not os.path.exists(input_fif):
        print("❌ ERROR: Input file does not exist!")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_fif), exist_ok=True)

    print(f"📂 Loading epoched EEG data from: {input_fif}")
    epochs = mne.read_epochs(input_fif, preload=True)

    print("🔄 Normalising data (channel-wise per epoch)...")
    epochs = normalise_epochs(epochs)

    print(f"💾 Saving normalised EEG data to: {output_fif}")
    epochs.save(output_fif, overwrite=True)
