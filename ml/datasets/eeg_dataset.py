import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# EEG Dataset class for when there is labels for epochs in the recording, i.e. each epoch has its own label that can be used for training
class EEGDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform

        # Load EEG data from HDF5
        with h5py.File(hdf5_file, "r") as f:
            self.data = np.array(f["EEG/data"])  # (n_epochs, n_channels, n_times)
            self.labels = np.array(f["EEG/labels"])  # (n_epochs,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            eeg_sample = self.transform(eeg_sample)

        return torch.tensor(eeg_sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# EEG Dataset class for when there is only one label for the whole recording, i.e. labelling is for whole recording, not for epochs in recording
class EEGRecordingDataset(Dataset):
    def __init__(self, hdf5_file, transform=None):
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.data = []
        self.labels = []

        # Load EEG recordings from HDF5
        with h5py.File(hdf5_file, "r") as f:
            self.recording_names = list(f["EEG"].keys())  # Get all recording names

            for rec_name in self.recording_names:
                eeg_data = np.array(f[f"EEG/{rec_name}/data"])  # Shape: (n_epochs, n_channels, n_times)
                label = np.array(f[f"EEG/{rec_name}/label"])  # Single label per recording

                # Concatenate all epochs into one 2D array (channels, timepoints)
                eeg_recording = eeg_data.reshape(eeg_data.shape[1], -1)  # Shape: (n_channels, total_timepoints)

                self.data.append(eeg_recording)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_recording = self.data[idx]  # (n_channels, total_timepoints)
        label = self.labels[idx]

        if self.transform:
            eeg_recording = self.transform(eeg_recording)

        return torch.tensor(eeg_recording, dtype=torch.float32), torch.tensor(label, dtype=torch.long)