#!/usr/bin/env python3

import sys
import os
import h5py
import mne
import yaml
import numpy as np

def convert_to_hdf5(input_fif, output_hdf5, config_file):
    """
    Convert an MNE epoched .fif file into an HDF5 format for model training.

    Parameters:
    - input_fif (str): Path to the normalized, epoched EEG .fif file.
    - output_hdf5 (str): Path to save the output HDF5 file.
    - config_file (str): Path to YAML config file with settings.
    """
    print("🔄 Loading config file...")
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    dataset_name = config.get("dataset_name", "EEG")
    chunk_size = config.get("chunk_size", 50)  # Define chunk size for HDF5 storage

    print(f"📂 Loading EEG epoched data from: {input_fif}")
    epochs = mne.read_epochs(input_fif, preload=True)

    # Convert epoched data to NumPy array
    eeg_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

    # Extract metadata
    sfreq = epochs.info["sfreq"]  # Sampling frequency
    channel_names = epochs.ch_names
    event_ids = epochs.events[:, -1]  # Extract event labels (if available)
    subject_id = os.path.basename(input_fif).split(".")[0]

    print(f"🔍 Data shape: {eeg_data.shape} (Epochs x Channels x Timepoints)")

    # Create HDF5 file
    print(f"💾 Saving to HDF5: {output_hdf5}")
    with h5py.File(output_hdf5, "w") as f:
        # Store EEG data
        f.create_dataset(f"{dataset_name}/data", data=eeg_data, chunks=(chunk_size, eeg_data.shape[1], eeg_data.shape[2]))
        
        # Store sampling frequency
        f.create_dataset(f"{dataset_name}/sfreq", data=sfreq)
        
        # Store channel names
        f.create_dataset(f"{dataset_name}/channel_names", data=np.array(channel_names, dtype='S'))

        # Store event labels if they exist
        if len(event_ids) == eeg_data.shape[0]:  # Ensure correct label count
            f.create_dataset(f"{dataset_name}/labels", data=event_ids)

        # Store subject ID as an attribute
        f.attrs["subject_id"] = subject_id

    print("✅ HDF5 conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/convert_to_hdf5.py <input_fif> <output_hdf5> <config_file>", flush=True)
        sys.exit(1)

    input_fif = sys.argv[1]
    output_hdf5 = sys.argv[2]
    config_file = sys.argv[3]

    convert_to_hdf5(input_fif, output_hdf5, config_file)
