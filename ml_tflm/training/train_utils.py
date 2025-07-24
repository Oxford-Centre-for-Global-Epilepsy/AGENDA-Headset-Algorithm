import h5py
import random
from ml_tflm.dataset.eeg_dataset import EEGRecordingTFGenerator
from ml_tflm.dataset.eeg_dataset_rewrite import EEGRecordingDatasetTF
import numpy as np

import json
import os

import tensorflow as tf

from collections import Counter


def load_label_config(json_path):
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

    return {
        "label_map": label_map,
        "inverse_label_map": inverse_label_map,
        "label_prior": label_prior
    }

def load_eeg_datasets_split(h5_file_path, dataset_name, label_config, 
                            val_frac=0.2, test_frac=0.1, k_fold=False,
                            omit_channels=None, seed=42, 
                            batch_size=1, shuffle=True, mirror_flag=False):
    # Load subject IDs
    with h5py.File(h5_file_path, "r") as f:
        subject_ids = sorted(list(f[dataset_name].keys()))

    # Shuffle subject IDs
    random.seed(seed)
    random.shuffle(subject_ids)
    n_total = len(subject_ids)

    # Compute split sizes
    n_test = round(n_total * test_frac)
    test_ids = subject_ids[:n_test]
    remaining_ids = subject_ids[n_test:]
    n_remaining = len(remaining_ids)

    if not k_fold:
        n_val = round(n_remaining * val_frac)
        val_ids = remaining_ids[:n_val]
        train_ids = remaining_ids[n_val:]

        if len(train_ids) == 0 or len(val_ids) == 0:
            raise ValueError("Not enough data for training or validation split. "
                             f"{len(train_ids)=}, {len(val_ids)=}, {n_total=}")
        
        # Generate the train validation sets
        train_gen = _make_tf_generator(train_ids, h5_file_path, dataset_name, label_config, omit_channels, mirror_flag=mirror_flag)
        train_val_sets = [(
            train_gen.get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1),
            _make_tf_generator(val_ids, h5_file_path, dataset_name, label_config, omit_channels).get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1)
            ),]
        
        # Get the label histograms
        label_histograms = [train_gen.get_label_histogram(),]

    else:
        # K-Fold mode
        n_val = round(n_remaining * val_frac)
        fold_size = max(1, n_val)
        num_folds = max(2, n_remaining // fold_size)

        # Interleaved fold assignment
        folds = [[] for _ in range(num_folds)]
        for idx, sid in enumerate(remaining_ids):
            folds[idx % num_folds].append(sid)

        train_val_sets = []
        label_histograms = []
        for i in range(num_folds):
            val_ids = folds[i]
            train_ids = [sid for j, fold in enumerate(folds) if j != i for sid in fold]

            if len(train_ids) == 0 or len(val_ids) == 0:
                raise ValueError(f"Fold {i}: empty training or validation set.")

            # Generate the train validation sets
            train_gen = _make_tf_generator(train_ids, h5_file_path, dataset_name, label_config, omit_channels, mirror_flag=mirror_flag)
            train_val_set = (
                train_gen.get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1),
                _make_tf_generator(val_ids, h5_file_path, dataset_name, label_config, omit_channels).get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1)
                )
            
            train_val_sets.append(train_val_set)

            # Get and append the label histograms
            label_histograms.append(train_gen.get_label_histogram())
            
            print(f"Dataset Splitter: {num_folds}-fold dataset generated")

    # Generate the test sets
    test_dataset =_make_tf_generator(test_ids, h5_file_path, dataset_name, label_config, omit_channels).get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1) if n_test > 0 else None
    
    return train_val_sets, test_dataset, label_histograms

def _make_tf_generator(subject_ids, h5_file_path, dataset_name, label_config, omit_channels, mirror_flag=False):
    generator = EEGRecordingDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        omit_channels=omit_channels,
        subject_ids=subject_ids,
        mirror_flag=mirror_flag
    )
    return generator

def get_model_size_tf(model):
    size_bytes = sum([tf.keras.backend.count_params(w) * w.dtype.size for w in model.weights])
    return size_bytes / 1e6  # MB

def get_checkpoint_manager(model, optimizer, ckpt_dir, max_to_keep=3):
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=ckpt_dir,
        max_to_keep=max_to_keep
    )
    return checkpoint, manager

def maybe_restore_checkpoint(checkpoint, manager):
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        return True
    else:
        return False

def compute_label_histogram(dataset, label_map_config):
    """
    Compute histogram of internal label indices from dataset and return string label counts.

    Args:
        dataset (tf.data.Dataset): Yields dicts with key "labels" of shape [B], already internal indices.
        label_map_config (dict): Contains 'label_map' (str → int)

    Returns:
        dict: {str_label: count}
    """
    label_map = label_map_config["label_map"]  # {str_label: hierarchical_index}
    
    # Step 1: Rebuild internal flat index map
    label_map_internal = {key: i for i, key in enumerate(label_map.keys())}
    inverse_label_map_internal = {v: k for k, v in label_map_internal.items()}

    counter = Counter()

    for batch in dataset:
        label_tensor = batch["internal_label"]  # shape [B], internal flat indices
        labels_np = label_tensor.numpy() if tf.is_tensor(label_tensor) else label_tensor

        for internal_idx in labels_np:
            label_str = inverse_label_map_internal[int(internal_idx)]
            counter[label_str] += 1

    return dict(counter)

def get_activation(name):
    return {
        "relu": tf.nn.relu,
        "leaky_relu": tf.nn.leaky_relu,
        "elu": tf.nn.elu,
        "selu": tf.nn.selu,
        "tanh": tf.nn.tanh,
        "sigmoid": tf.nn.sigmoid,
        "hard_sigmoid": tf.keras.activations.hard_sigmoid,
        "hard_swish": hard_swish,
        "gelu": tf.nn.gelu,
        "swish": tf.nn.swish
    }[name]

def hard_swish(x):
    return x * tf.nn.relu6(x + 3) / 6.0

def print_confusion_matrix(cm: np.ndarray, class_labels=None):
    """
    Pretty-print a confusion matrix to the terminal.

    Args:
        cm (np.ndarray): 2D array confusion matrix.
        class_labels (list of str): Optional class label names.
    """
    if class_labels is None:
        class_labels = [str(i) for i in range(cm.shape[0])]

    # Ensure same width for all labels
    max_label_len = max(len(label) for label in class_labels)
    col_width = max(max_label_len, 5)

    # Header
    print(" " * (col_width + 2) + "Predicted")
    print(" " * (col_width + 2) + "  ".join(f"{label:>{col_width}}" for label in class_labels))
    print(" " * (col_width + 2) + "-" * (col_width + 2) * len(class_labels))

    # Rows
    for i, row in enumerate(cm):
        row_label = class_labels[i]
        row_str = "  ".join(f"{val:>{col_width}d}" for val in row)
        print(f"{row_label:>{col_width}} | {row_str}")
