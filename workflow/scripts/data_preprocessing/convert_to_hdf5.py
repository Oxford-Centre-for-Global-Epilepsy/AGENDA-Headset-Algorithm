#!/usr/bin/env python3

import sys
import os
import h5py
import mne
import yaml
import numpy as np


def extract_class_labels(file_path, site):
    """
    Extract hierarchical class labels from the EEG file path based on site argument.

    Example:
    - "data/edf/india/epileptic/focal/left/MK535FH5.edf" -> ["epileptic", "focal", "left"]
    - "data/edf/india/epileptic/generalized/MK535FH5.edf" -> ["epileptic", "generalized"]
    - "data/edf/india/neurotypical/MK535FH5.edf" -> ["neurotypical"]

    Parameters:
    - file_path (str): Full path to the EEG file.
    - site (str): Site name (e.g., "india"), used to determine where labels begin.

    Returns:
    - class_labels (list): List of hierarchical class labels.
    """
    # Normalize path to ensure compatibility across OS
    file_path = os.path.normpath(file_path)

    # Split the path into components
    path_parts = file_path.split(os.sep)

    # Find the index of the site in the path
    try:
        site_index = path_parts.index(site)
    except ValueError:
        raise ValueError(f"❌ ERROR: Site '{site}' not found in file path: {file_path}")

    # Extract class labels (everything AFTER the site, EXCLUDING the filename)
    class_labels = path_parts[site_index + 1 : -1]

    return class_labels


def convert_to_hdf5(input_fif, output_hdf5, config_file, montage_type, montage_name, site, data_class_label):
    """
    Convert an MNE epoched .fif file into an HDF5 format for model training.

    Parameters:
    - input_fif (str): Path to the normalized, epoched EEG .fif file.
    - output_hdf5 (str): Path to save the output HDF5 file.
    - config_file (str): Path to YAML config file with settings.
    - montage_type (str): Montage type (raw, monopolar, bipolar).
    - montage_name (str): Name of the montage used.
    - site (str): Site name, used to extract class labels.
    - data_class_label (str): Class labels for the data.
    """
    print(f"🔄 Loading config file: {config_file}")
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
    epoch_duration = n_times / sfreq
    total_duration = n_epochs * epoch_duration  # Sum across all epochs

    print(f"✅ Data Loaded: {n_epochs} epochs, {n_channels} channels, {n_times} time points per epoch")
    print(f"⏳ Total data duration: {total_duration:.2f} seconds ({epoch_duration:.2f}s per epoch)")

    # Extract subject ID
    subject_id = os.path.basename(input_fif).split(".")[0]

    # Extract hierarchical class labels using site
    #class_labels = extract_class_labels(input_fif, site)
    class_labels = data_class_label.split("_")
    
    print(f"🏷️ Class Labels for {subject_id}: {class_labels}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_hdf5), exist_ok=True)

    # Define chunking strategy (chunk epochs in groups of 50, keep full channels & timepoints)
    epoch_chunk_size = min(chunk_size, n_epochs)
    chunk_shape = (epoch_chunk_size, n_channels, n_times)

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
        f.create_dataset(f"{dataset_name}/channel_names", data=np.array(epochs.ch_names, dtype='S'))

        # Store montage details
        f.attrs["montage_type"] = montage_type
        f.attrs["montage_name"] = montage_name

        # Store hierarchical class labels
        f.create_dataset(f"{dataset_name}/class_labels", data=np.array(class_labels, dtype='S'))
        f.attrs["class_label_hierarchy"] = " > ".join(class_labels)

        # Additional metadata
        f.attrs["subject_id"] = subject_id
        f.attrs["n_epochs"] = n_epochs
        f.attrs["n_channels"] = n_channels
        f.attrs["n_times"] = n_times
        f.attrs["chunk_shape"] = chunk_shape

    print("✅ HDF5 conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python scripts/convert_to_hdf5.py <input_fif> <output_hdf5> <config_file> <montage_type> <montage_name> <site> <class_label>", flush=True)
        sys.exit(1)

    input_fif = sys.argv[1]
    output_hdf5 = sys.argv[2]
    config_file = sys.argv[3]
    montage_type = sys.argv[4]
    montage_name = sys.argv[5]
    site = sys.argv[6]
    data_class_label = sys.argv[7]

    convert_to_hdf5(input_fif, output_hdf5, config_file, montage_type, montage_name, site, data_class_label)
