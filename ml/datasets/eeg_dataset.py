import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# EEG Recording Dataset (Expects that loaded data contains high-level label for each recording, i.e. not epoch-level labels)
class EEGRecordingDataset(Dataset):
    def __init__(self, h5_file_path, dataset_name, label_map=None, transform=None, omit_channels=None, subject_ids=None):
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
            omit_channels (list of str): Channel names to omit during loading
            subject_ids (list of str): Subject ids to include in the dataset during loading (to faciliate train/validation/test dataset specific loading)

        """
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.label_map = label_map or {}
        self.transform = transform
        self.omit_channels = set(omit_channels or [])

        with h5py.File(self.h5_file_path, 'r') as f:
            
            # Load only the specified subject ids
            all_subject_ids = list(f[self.dataset_name].keys())
            self.subject_ids = subject_ids if subject_ids is not None else all_subject_ids
            
            # The max number of epochs in the dataset (for padding out some of the recordings)
            self.max_epochs = f.attrs["max_epochs"]
            
            # Get the channel names
            first_subject = self.subject_ids[0]
            channel_names_raw = f[self.dataset_name][first_subject].attrs["channel_names"]
            self.original_channel_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in channel_names_raw]

            # Compute keep indices (channels to retain)
            self.keep_indices = [i for i, ch in enumerate(self.original_channel_names) if ch not in self.omit_channels]
            self.channel_names = [self.original_channel_names[i] for i in self.keep_indices]

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]

        with h5py.File(self.h5_file_path, 'r') as f:
            subj_group = f[self.dataset_name][subject_id]
            data = subj_group["data"][()]  # shape: [epochs, channels, time]
            
            # Subselect channels if needed
            if self.keep_indices:
                data = data[:, self.keep_indices, :]

            # Get the class labels and decode bytes if needed
            labels = subj_group.attrs["class_labels"]
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
            n_epochs, _, n_time = data.shape
            epoch_mask = np.ones(self.max_epochs, dtype=np.bool_)
            if n_epochs < self.max_epochs:
                pad = np.zeros((self.max_epochs - n_epochs, len(self.keep_indices), n_time), dtype=data.dtype)
                data = np.concatenate([data, pad], axis=0)
                epoch_mask[n_epochs:] = 0
            elif n_epochs > self.max_epochs:
                data = data[:self.max_epochs]

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
        """Returns the list of channel names"""
        return self.channel_names

    def get_omitted_channel_names(self):
        """Returns the list of omitted channel names"""
        return self.omit_channels

    def get_num_channels(self):
        """Returns the number of EEG channels after omission."""
        return len(self.keep_indices)

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

    def get_target_labels(self):
        """
        Returns a list of all the target labels across the entire dataset.
        Returns a list of tuples: [(level1, level2, level3), ...]
        """
        target_labels = []
        with h5py.File(self.h5_file_path, "r") as f:
            group = f[self.dataset_name]
            for sid in group.keys():
                labels = group[sid].attrs["class_labels"]
                if isinstance(labels[0], bytes):
                    labels = [l.decode("utf-8") for l in labels]
                # Collect all 3 levels of labels for each subject
                label_tuple = (
                    self.label_map.get(labels[0], -1),
                    self.label_map.get(labels[1], -1) if len(labels) > 1 else -1,
                    self.label_map.get(labels[2], -1) if len(labels) > 2 else -1
                )
                target_labels.append(label_tuple)
        return target_labels

    def get_class_weights(self):
        """
        Calculates balanced class weights for each classification level (level 1, level 2, and level 3).
        Returns the class weights as PyTorch tensors.
        """
        # Get all labels from the dataset
        all_labels = self.get_target_labels()

        # Separate labels by level
        level1_labels = [label[0] for label in all_labels]  # Labels from level 1
        level2_labels = [label[1] for label in all_labels]  # Labels from level 2
        level3_labels = [label[2] for label in all_labels]  # Labels from level 3

        # Remove `-1` values (masked labels) for each level
        level1_labels = [label for label in level1_labels if label != -1]
        level2_labels = [label for label in level2_labels if label != -1]
        level3_labels = [label for label in level3_labels if label != -1]

        # Compute class weights for each level independently
        level1_class_weights = class_weight.compute_class_weight('balanced', 
                                                                  classes=np.unique(level1_labels), 
                                                                  y=level1_labels)

        level2_class_weights = class_weight.compute_class_weight('balanced', 
                                                                  classes=np.unique(level2_labels), 
                                                                  y=level2_labels)

        level3_class_weights = class_weight.compute_class_weight('balanced', 
                                                                  classes=np.unique(level3_labels), 
                                                                  y=level3_labels)

        # Convert to tensor for PyTorch compatibility
        level1_class_weights_tensor = torch.tensor(level1_class_weights, dtype=torch.float)
        level2_class_weights_tensor = torch.tensor(level2_class_weights, dtype=torch.float)
        level3_class_weights_tensor = torch.tensor(level3_class_weights, dtype=torch.float)

        # Return class weights for each level
        return level1_class_weights_tensor, level2_class_weights_tensor, level3_class_weights_tensor
