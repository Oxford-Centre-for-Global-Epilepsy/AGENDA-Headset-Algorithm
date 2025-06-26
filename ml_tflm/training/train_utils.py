import h5py
import random
from ml_tflm.dataset.eeg_dataset import EEGRecordingTFGenerator
import numpy as np

import json
import os

import tensorflow as tf

def load_label_config_with_interpretation(json_path):
    """
    Load label_map and label_prior from a JSON file and generate useful interpretation mappings.

    Args:
        json_path (str): Path to the JSON config file.

    Returns:
        dict: {
            "label_map": forward label map,
            "inverse_label_map": inverse of label_map,
            "probabilistic_vector_map": dict mapping label string to soft class vector
        }
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Label config file not found at: {json_path}")
    
    with open(json_path, "r") as f:
        config = json.load(f)

    label_map = config.get("label_map")
    label_prior = config.get("label_prior")

    if not label_map or not label_prior:
        raise ValueError("Both 'label_map' and 'label_prior' must exist in JSON config.")

    inverse_label_map = {v: k for k, v in label_map.items()}

    # Build probabilistic_vector_map
    probabilistic_vector_map = {}
    class_map = {
        "neurotypical": 0,
        "generalized": 1,
        "left": 2,
        "right": 3
    }

    for label, idx in label_map.items():
        vec = np.zeros(len(class_map), dtype=np.float32)

        if label == "neurotypical":
            vec[class_map["neurotypical"]] = 1.0
        elif label == "generalized":
            # epileptic + not focal → generalized
            vec[class_map["generalized"]] = label_prior["epileptic"] * (1 - label_prior["focal"])
        elif label == "left":
            # epileptic + focal + left
            vec[class_map["left"]] = label_prior["epileptic"] * label_prior["focal"] * label_prior["left"]
        elif label == "right":
            # epileptic + focal + right
            vec[class_map["right"]] = label_prior["epileptic"] * label_prior["focal"] * (1 - label_prior["left"])
        elif label == "epileptic":
            # Unknown type → use all subtypes based on prior
            vec[class_map["generalized"]] = label_prior["epileptic"] * (1 - label_prior["focal"])
            vec[class_map["left"]] = label_prior["epileptic"] * label_prior["focal"] * label_prior["left"]
            vec[class_map["right"]] = label_prior["epileptic"] * label_prior["focal"] * (1 - label_prior["left"])
        elif label == "focal":
            # Unknown laterality → split focal between left/right
            vec[class_map["left"]] = label_prior["epileptic"] * label_prior["focal"] * label_prior["left"]
            vec[class_map["right"]] = label_prior["epileptic"] * label_prior["focal"] * (1 - label_prior["left"])
        else:
            raise ValueError(f"Unexpected label: {label}")

        # Normalize for safety
        total = vec.sum()
        if total > 0:
            vec /= total
        probabilistic_vector_map[label] = vec

    return {
        "label_map": label_map,
        "inverse_label_map": inverse_label_map,
        "probabilistic_vector_map": probabilistic_vector_map
    }

def load_eeg_datasets_split(h5_file_path, dataset_name, label_map=None,
                             omit_channels=None, seed=42,
                             train_frac=0.7, val_frac=0.2, test_frac=0.1,
                             batch_size=1, shuffle=True):
    """
    Splits EEG dataset into train/val/test TensorFlow datasets by subject ID.

    Args:
        h5_file_path (str): Path to the HDF5 file containing EEG data.
        dataset_name (str): Root group name in the HDF5 file.
        label_map (dict, optional): Maps string labels to integer IDs.
        omit_channels (list[str], optional): Channel names to exclude.
        seed (int): Random seed for reproducibility.
        train_frac (float): Fraction of data for training set.
        val_frac (float): Fraction for validation set.
        test_frac (float): Fraction for test set.
        batch_size (int): Batch size for TF datasets.
        shuffle (bool): Whether to shuffle dataset during loading.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset) as tf.data.Dataset
    """

    # Ensure fractions sum to 1
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    # Read subject IDs from HDF5 file
    with h5py.File(h5_file_path, "r") as f:
        subject_ids = sorted(list(f[dataset_name].keys()))

    # Shuffle subject IDs deterministically
    random.seed(seed)
    random.shuffle(subject_ids)

    # Compute split indices
    n = len(subject_ids)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Partition subject IDs
    train_ids = subject_ids[:train_end]
    val_ids = subject_ids[train_end:val_end]
    test_ids = subject_ids[val_end:]

    # Helper to wrap subject IDs in a TF-compatible generator
    def make_tf_dataset(subject_ids):
        generator = EEGRecordingTFGenerator(
            h5_file_path=h5_file_path,
            dataset_name=dataset_name,
            label_map=label_map,
            omit_channels=omit_channels,
            subject_ids=subject_ids
        )
        return generator.as_dataset(batch_size=batch_size, shuffle=shuffle)

    return (
        make_tf_dataset(train_ids),
        make_tf_dataset(val_ids),
        make_tf_dataset(test_ids)
    )

# TODO: Add interpretor function for soft class vectors
