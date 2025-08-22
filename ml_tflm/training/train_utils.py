import h5py
import random
from ml_tflm.dataset.eeg_dataset_rewrite import EEGRecordingDatasetTF
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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

def prepare_eeg_datasets(
    h5_file_path: str,
    dataset_name: str,
    label_config: dict,
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    k_fold: bool = False,
    stratify: bool = False,
    seed: int = 42,
    omit_channels: Optional[List[str]] = None,
    batch_size: int = 1,
    shuffle: bool = True,
    mirror_flag: bool = False,
    internal_label_cap: Optional[Dict[str, int]] = None,
    chunk_size: Optional[int] = None
) -> Tuple[List[Tuple[tf.data.Dataset, tf.data.Dataset]], Optional[tf.data.Dataset], List[dict]]:
    """
    Prepares EEG datasets by splitting subject metadata and generating tf.data.Dataset objects.
    Supports optional stratification by 'internal_label' and optional class-wise segment cap.
    """
    dummy_dataset = EEGRecordingDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
    )
    sample_metadata = dummy_dataset.sample_metadata
    dummy_dataset.close()

    if internal_label_cap:
        # Group metadata by label
        grouped = defaultdict(list)
        for entry in sample_metadata:
            grouped[entry["internal_label"]].append(entry)

        # Truncate each group if needed
        sample_metadata = []
        for label, entries in grouped.items():
            if label in internal_label_cap:
                # Sort to ensure reproducibility
                random.Random(seed).shuffle(entries)
                sample_metadata.extend(entries[:internal_label_cap[label]])
            else:
                sample_metadata.extend(entries)

    if stratify:
        grouped = defaultdict(list)
        for entry in sample_metadata:
            grouped[entry["internal_label"]].append(entry)

        # Accumulate results from per-group splits
        train_id_lists_by_group = []
        val_id_lists_by_group = []
        test_ids_all = []

        for group in grouped.values():
            train_lists, val_lists, test_ids = split_subjects(
                sample_metadata=group,
                val_frac=val_frac,
                test_frac=test_frac,
                seed=seed,
                k_fold=k_fold,
            )
            train_id_lists_by_group.append(train_lists)
            val_id_lists_by_group.append(val_lists)
            test_ids_all.extend(test_ids)

        num_folds = len(train_id_lists_by_group[0])
        train_id_lists = [[] for _ in range(num_folds)]
        val_id_lists = [[] for _ in range(num_folds)]

        for fold_idx in range(num_folds):
            for group_train_lists in train_id_lists_by_group:
                train_id_lists[fold_idx].extend(group_train_lists[fold_idx])
            for group_val_lists in val_id_lists_by_group:
                val_id_lists[fold_idx].extend(group_val_lists[fold_idx])

        test_ids = test_ids_all

    else:
        train_id_lists, val_id_lists, test_ids = split_subjects(
            sample_metadata=sample_metadata,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            k_fold=k_fold,
        )

    train_val_sets, test_dataset, label_histograms = generate_tf_datasets(
        train_id_lists=train_id_lists,
        val_id_lists=val_id_lists,
        test_ids=test_ids,
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        omit_channels=omit_channels,
        mirror_flag=mirror_flag,
        batch_size=batch_size,
        shuffle=shuffle,
        chunk_size=chunk_size
    )

    return train_val_sets, test_dataset, label_histograms

def split_subjects(
    sample_metadata: List[Dict],
    val_frac: float = 0.2,
    test_frac: float = 0.1,
    seed: int = 42,
    k_fold: bool = False,
) -> Tuple[List[List[str]], List[List[str]], List[str]]:
    """
    Splits subject metadata into train, val, and test sets.

    Returns:
        - train_id_lists: list of subject ID lists for training (1 or k-fold)
        - val_id_lists: list of subject ID lists for validation (1 or k-fold)
        - test_ids: list of subject IDs for testing
    """
    random.seed(seed)
    shuffled = sample_metadata[:]
    random.shuffle(shuffled)

    subject_ids = [entry["subject_id"] for entry in shuffled]

    n_total = len(subject_ids)
    n_test = round(n_total * test_frac)

    test_ids = subject_ids[:n_test]
    remaining_ids = subject_ids[n_test:]

    if not k_fold:
        n_val = round(len(remaining_ids) * val_frac)
        val_ids = remaining_ids[:n_val]
        train_ids = remaining_ids[n_val:]
        return [train_ids], [val_ids], test_ids

    # K-fold logic
    num_folds = max(2, round(1 / val_frac))
    folds = [[] for _ in range(num_folds)]
    for idx, sid in enumerate(remaining_ids):
        folds[idx % num_folds].append(sid)

    train_id_lists = []
    val_id_lists = []
    for i in range(num_folds):
        val_ids = folds[i]
        train_ids = [sid for j, fold in enumerate(folds) if j != i for sid in fold]
        train_id_lists.append(train_ids)
        val_id_lists.append(val_ids)

    return train_id_lists, val_id_lists, test_ids

def generate_tf_datasets(
    train_id_lists: List[List[str]],
    val_id_lists: List[List[str]],
    test_ids: List[str],
    h5_file_path: str,
    dataset_name: str,
    label_config: dict,
    omit_channels: Optional[List[str]] = None,
    mirror_flag: bool = False,
    batch_size: int = 1,
    shuffle: bool = True,
    chunk_size: Optional[int] = None

) -> Tuple[List[Tuple[tf.data.Dataset, tf.data.Dataset]], Optional[tf.data.Dataset], List[dict]]:
    """
    Generates TensorFlow datasets from subject ID splits.
    """
    train_val_sets = []
    label_histograms = []

    for train_ids, val_ids in zip(train_id_lists, val_id_lists):
        train_gen = _make_tf_generator(train_ids, h5_file_path, dataset_name,
                                       label_config, omit_channels, mirror_flag=mirror_flag, chunk_size=chunk_size)
        val_gen = _make_tf_generator(val_ids, h5_file_path, dataset_name,
                                     label_config, omit_channels, chunk_size=chunk_size, deterministic_draw=True)

        train_dataset = train_gen.get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1).repeat()
        val_dataset = val_gen.get_tf_dataset(batch_size=batch_size, shuffle=shuffle, num_parallel_calls=1)

        train_val_sets.append((train_dataset, val_dataset))
        label_histograms.append(train_gen.get_label_histogram())

    # Test dataset
    test_dataset = None
    if test_ids:
        # The test set will always be full (un-truncated) recordings
        test_gen = _make_tf_generator(test_ids, h5_file_path, dataset_name,
                                      label_config, omit_channels, chunk_size=None)
        test_dataset = test_gen.get_tf_dataset(batch_size=1, shuffle=shuffle, num_parallel_calls=1)

    return train_val_sets, test_dataset, label_histograms

def _make_tf_generator(subject_ids, h5_file_path, dataset_name, label_config, omit_channels, mirror_flag=False, chunk_size=None, deterministic_draw=False):
    generator = EEGRecordingDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        omit_channels=omit_channels,
        subject_ids=subject_ids,
        mirror_flag=mirror_flag,
        chunk_size=chunk_size,
        deterministic_draw=deterministic_draw
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

def clean_metrics(metrics):
    if isinstance(metrics, list):
        return [clean_metrics(m) for m in metrics]
    elif isinstance(metrics, dict):
        return {
            k: (
                v.tolist() if isinstance(v, np.ndarray)
                else [float(x) for x in v] if isinstance(v, list) and all(isinstance(x, (np.floating, np.float32, np.float64)) for x in v)
                else float(v) if isinstance(v, (np.floating, np.float32, np.float64))
                else v
            )
            for k, v in metrics.items()
        }
    return metrics

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
