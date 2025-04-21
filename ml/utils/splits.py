from collections import defaultdict
import random
from ml.datasets.eeg_dataset import EEGRecordingDataset

def stratified_split_subjects(subject_to_labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Performs stratified split based on label triplets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    random.seed(seed)

    label_groups = defaultdict(list)
    for sid, labels in subject_to_labels.items():
        label_groups[tuple(labels)].append(sid)

    splits = {"train": [], "val": [], "test": []}
    for label_triplet, sids in label_groups.items():
        random.shuffle(sids)
        n = len(sids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(sids[:n_train])
        splits["val"].extend(sids[n_train:n_train + n_val])
        splits["test"].extend(sids[n_train + n_val:])
    return splits

def create_stratified_datasets(h5_path, dataset_name, label_map, ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Creates stratified train/val/test EEGRecordingDataset objects.

    Args:
        h5_path (str): Path to combined .h5 file
        dataset_name (str): Name of root group (e.g. 'combined_monopolar')
        label_map (dict): Class label mapping
        ratios (tuple): (train_ratio, val_ratio, test_ratio)
        seed (int): Random seed

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    full_dataset = EEGRecordingDataset(h5_path, dataset_name, label_map)
    subject_to_labels = full_dataset.get_subjects_with_labels()

    splits = stratified_split_subjects(subject_to_labels, *ratios, seed=seed)

    def make_filtered_dataset(split_ids):
        ds = EEGRecordingDataset(h5_path, dataset_name, label_map)
        ds.filter_subjects(split_ids)
        return ds

    return (
        make_filtered_dataset(splits["train"]),
        make_filtered_dataset(splits["val"]),
        make_filtered_dataset(splits["test"])
    )

def create_kfold_stratified_datasets(h5_path, dataset_name, label_map, k_folds, fold_index, seed=42):
    """
    Creates train/val/test datasets using stratified K-Fold splitting.

    Args:
        h5_path (str)
        dataset_name (str)
        label_map (dict)
        k_folds (int)
        fold_index (int)
        seed (int)

    Returns:
        train_dataset, val_dataset, test_dataset
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    from ml.datasets.eeg_dataset import EEGRecordingDataset

    full_dataset = EEGRecordingDataset(h5_path, dataset_name, label_map)
    subject_to_labels = full_dataset.get_subjects_with_labels()
    subjects = list(subject_to_labels.keys())
    labels = [tuple(subject_to_labels[s]) for s in subjects]

    # Map label triplets to string for stratification
    labels_str = ["|".join(l) for l in labels]

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(subjects, labels_str))
    train_ids, val_ids = splits[fold_index]

    train_subjects = [subjects[i] for i in train_ids]
    val_subjects = [subjects[i] for i in val_ids]

    def make_filtered(ids):
        ds = EEGRecordingDataset(h5_path, dataset_name, label_map)
        ds.filter_subjects(ids)
        return ds

    return make_filtered(train_subjects), make_filtered(val_subjects),  make_filtered(val_subjects)  # test and val sets are the same here - need to consider handling them separately for final performance evaluation
