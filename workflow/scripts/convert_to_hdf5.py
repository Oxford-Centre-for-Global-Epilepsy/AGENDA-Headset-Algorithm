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
    n_epochs, n_channels, n_times = eeg_data.shape

    # Compute total duration: each epoch has the same duration (n_times / sfreq)
    epoch_duration = n_times / sfreq
    total_duration = n_epochs * epoch_duration  # Sum across all epochs
    
    print(f"✅ Data Loaded: {n_epochs} epochs, {n_channels} channels, {n_times} time points per epoch")
    print(f"⏳ Total data duration: {total_duration:.2f} seconds ({epoch_duration:.2f}s per epoch)")

    # Get channel names
    channel_names = epochs.ch_names

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)

    # Define chunking strategy (chunk epochs in groups of 50, keep full channels & timepoints)
    epoch_chunk_size = min(50, n_epochs)  # Chunk size of 50 epochs (or all if fewer than 50)
    chunk_shape = (epoch_chunk_size, n_channels, n_times)

    # Get the subject id 
    subject_id = os.path.basename(input_fif).split(".")[0]

    print(f"🔍 Data shape: {eeg_data.shape} (Epochs x Channels x Timepoints)")

    # Create HDF5 file
    print(f"💾 Saving to HDF5: {output_hdf5}")
    with h5py.File(output_hdf5, "w") as f:
        
        # Store EEG data
        f.create_dataset(f"{dataset_name}/data", data=eeg_data, chunks=chunk_shape, compression="gzip", compression_opts=4)
        
        # Store sampling frequency
        f.create_dataset(f"{dataset_name}/sfreq", data=sfreq)
        
        # Store the total duration of the EEG dataset
        f.create_dataset(f"{dataset_name}/total_duration", data=total_duration)  # Total duration of the dataset
        
        # Store the duration of epochs in the EEG dataset
        f.create_dataset(f"{dataset_name}/epoch_duration", data=epoch_duration)  # Duration of each epoch
    
        # Store channel names
        f.create_dataset(f"{dataset_name}/channel_names", data=np.array(channel_names, dtype='S'))

        # Store metadata attributes for the dataset
        f.attrs["subject_id"] = subject_id
        f.attrs["n_epochs"] = n_epochs
        f.attrs["n_channels"] = n_channels
        f.attrs["n_times"] = n_times
        f.attrs["chunk_shape"] = chunk_shape

    print("✅ HDF5 conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/convert_to_hdf5.py <input_fif> <output_hdf5> <config_file>", flush=True)
        sys.exit(1)

    input_fif = sys.argv[1]
    output_hdf5 = sys.argv[2]
    config_file = sys.argv[3]

    convert_to_hdf5(input_fif, output_hdf5, config_file)
