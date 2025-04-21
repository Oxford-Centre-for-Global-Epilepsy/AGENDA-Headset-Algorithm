import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# EEG Recording Dataset (Expects that loaded data contains high-level label for each recording, i.e. not epoch-level labels)
class EEGRecordingDataset(Dataset):
    def __init__(self, h5_file_path, dataset_name, label_map=None, transform=None):
        """
        Args:
            h5_file_path (str): Path to combined HDF5 file.
            dataset_name (str): Root group in HDF5 (e.g. 'my_dataset').
            label_map (dict): Mapping from string labels to integer class IDs, e.g.:
                              {
                                'neurotypical': 0,
                                'epileptic': 1,
                                'focal': 2,
                                'generalized': 3,
                                'left': 4,
                                'right': 5
                              }
            transform (callable): Optional transform to apply to the EEG data.
        """
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.label_map = label_map or {}
        self.transform = transform

        with h5py.File(self.h5_file_path, 'r') as f:
            self.subject_ids = list(f[self.dataset_name].keys())
            self.max_epochs = f.attrs["max_epochs"]
            
            # Get the channel names
            first_subject = self.subject_ids[0]
            channel_names_raw = f[self.dataset_name][first_subject].attrs["channel_names"]
            self.channel_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in channel_names_raw]

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        with h5py.File(self.h5_file_path, 'r') as f:
            subj_group = f[self.dataset_name][subject_id]
            data = subj_group["data"][()]  # shape: [epochs, channels, time]
            labels = subj_group.attrs["class_labels"]

            # Decode bytes if needed
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]

            # Parse labels with padding to 3 levels and create a mask
            label_ids = []
            label_mask = []

            # Level 1: always exists
            label_ids.append(self.label_map.get(labels[0], -1))
            label_mask.append(1)

            # Level 2: optional
            if len(labels) > 1:
                label_ids.append(self.label_map.get(labels[1], -1))
                label_mask.append(1)
            else:
                label_ids.append(-1)
                label_mask.append(0)

            # Level 3: optional
            if len(labels) > 2:
                label_ids.append(self.label_map.get(labels[2], -1))
                label_mask.append(1)
            else:
                label_ids.append(-1)
                label_mask.append(0)

            # Pad or truncate EEG epochs
            n_epochs, n_channels, n_time = data.shape
            epoch_mask = np.ones(self.max_epochs, dtype=np.bool_)
            if n_epochs < self.max_epochs:
                pad = np.zeros((self.max_epochs - n_epochs, n_channels, n_time), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
                epoch_mask[n_epochs:] = 0
            elif n_epochs > self.max_epochs:
                data = data[:self.max_epochs]

            # Apply optional transform
            if self.transform:
                data = self.transform(data)

        return {
            "data": torch.tensor(data, dtype=torch.float32),               # [E, C, T]
            "labels": torch.tensor(label_ids, dtype=torch.long),           # [3]
            "label_mask": torch.tensor(label_mask, dtype=torch.bool),      # [3]
            "attention_mask": torch.tensor(epoch_mask, dtype=torch.bool),  # [E]
            "subject_id": subject_id

        }
    
    def get_channel_names(self):
        return self.channel_names

    def get_subject_ids(self):
        """Returns list of subject IDs in the dataset."""
        return self.subject_ids

    def get_subjects_with_labels(self):
        """
        Returns a dict: {subject_id: (level1, level2, level3)} label triplets
        """
        label_map = {}
        with h5py.File(self.h5_file_path, "r") as f:
            group = f[self.dataset_name]
            for sid in group.keys():
                labels = group[sid].attrs["class_labels"]
                if isinstance(labels[0], bytes):
                    labels = [l.decode("utf-8") for l in labels]
                label_map[sid] = tuple(labels)
        return label_map

    def filter_subjects(self, include_ids):
        """Filters dataset to only include specified subject IDs."""
        self.subject_ids = [sid for sid in self.subject_ids if sid in include_ids]

# Old Code to update: EEG Dataset class for when there is labels for epochs in the recording, i.e. each epoch has its own label that can be used for training
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
