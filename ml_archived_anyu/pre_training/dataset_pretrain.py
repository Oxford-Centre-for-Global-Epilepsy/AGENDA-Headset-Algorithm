import h5py
import numpy as np
import tensorflow as tf
import atexit
import random

class EEGContrastiveBufferDatasetTF:
    def __init__(self, h5_file_path, dataset_name, label_config,
                 max_epochs=300, buffer_size=32, batch_size=16,
                 omit_channels=None, transform=None, subject_ids=None):
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.label_map = label_config["label_map"]
        self.inverse_label_map = label_config["inverse_label_map"]
        self.label_map_internal = {
            "neurotypical": 0,
            "generalized": 1,
            "left": 2,
            "right": 2,
        }

        self.num_classes = len(set(self.label_map_internal.values()))  # or hardcode 4 if constant

        print("Number of classes:", self.num_classes)
        print("Label map:", self.label_map_internal)


        self.max_epochs = max_epochs
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.omit_channels = set(omit_channels or [])
        self.transform = transform

        self.h5 = h5py.File(h5_file_path, "r", swmr=True)
        atexit.register(self.close)

        all_subject_ids = list(self.h5[dataset_name].keys())
        self.subject_ids_all = subject_ids if subject_ids is not None else all_subject_ids

        self.buffer_subjects = []
        self.buffer_epochs = []
        self.buffer_labels = []
        self.channel_names = self._init_channels()

        self.class_subject_map = self._map_subjects_to_classes()
        self._reload_buffer()

    def _init_channels(self):
        first_subject = self.subject_ids_all[0]
        raw_names = self.h5[self.dataset_name][first_subject].attrs["channel_names"]
        names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in raw_names]
        self.keep_indices = [i for i, ch in enumerate(names) if ch not in self.omit_channels]
        return [names[i] for i in self.keep_indices]

    def _map_subjects_to_classes(self):
        class_subject_map = {c: [] for c in range(self.num_classes)}

        for subject_id in self.subject_ids_all:
            grp = self.h5[self.dataset_name][subject_id]
            labels = grp.attrs["class_labels"]
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]

            for i in reversed(range(3)):
                if i < len(labels) and labels[i] in self.label_map_internal:
                    class_id = int(self.label_map_internal[labels[i]])
                    class_subject_map[class_id].append(subject_id)
                    break

        return class_subject_map

    def _reload_buffer(self):
        self.buffer_subjects = random.sample(self.subject_ids_all, self.buffer_size)
        self.buffer_epochs = []
        self.buffer_labels = []

        for subject_id in self.buffer_subjects:
            grp = self.h5[self.dataset_name][subject_id]
            data = grp["data"][()]
            if self.keep_indices:
                data = data[:, self.keep_indices, :]

            labels = grp.attrs["class_labels"]
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]

            for i in reversed(range(3)):
                if i < len(labels) and labels[i] in self.label_map_internal:
                    internal_label = int(self.label_map_internal[labels[i]])
                    break
            else:
                continue  # skip if invalid label

            mask = np.ones((data.shape[0],), dtype=bool)
            if "epoch_mask" in grp:
                mask = grp["epoch_mask"][()].astype(bool)
                mask = mask[:data.shape[0]]

            for i in range(data.shape[0]):
                if not mask[i]:
                    continue
                self.buffer_epochs.append(data[i])
                self.buffer_labels.append(internal_label)

        self.buffer_epochs = np.stack(self.buffer_epochs)
        self.buffer_labels = np.array(self.buffer_labels, dtype=np.int32)

    def generator(self):
        while True:
            idxs = np.random.choice(len(self.buffer_epochs), self.batch_size, replace=False)
            batch_data = self.buffer_epochs[idxs]
            batch_labels = self.buffer_labels[idxs]

            batch_data = batch_data.astype(np.float32)
            batch_data = batch_data[..., np.newaxis]  # Add channel dim
            yield {"data": batch_data, "internal_label": batch_labels}

    def get_tf_dataset(self):
        output_signature = {
            "data": tf.TensorSpec(shape=(self.batch_size, len(self.keep_indices), None, 1), dtype=tf.float32),
            "internal_label": tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32)
        }
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def close(self):
        if self.h5:
            self.h5.close()
            self.h5 = None

class EEGPatientBatchDatasetTF:
    def __init__(self, h5_file_path, dataset_name, label_config,
                 batch_size=4, omit_channels=None, transform=None,
                 subject_ids=None):
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.label_map = label_config["label_map"]
        self.inverse_label_map = label_config["inverse_label_map"]
        self.label_map_internal = {
            "neurotypical": 0,
            "generalized": 1,
            "left": 2,
            "right": 3,
        }
        self.batch_size = batch_size
        self.omit_channels = set(omit_channels or [])
        self.transform = transform

        self.h5 = h5py.File(h5_file_path, "r", swmr=True)
        atexit.register(self.close)

        all_subject_ids = list(self.h5[dataset_name].keys())
        self.subject_ids_all = subject_ids if subject_ids is not None else all_subject_ids

        self.channel_names = self._init_channels()

        self.max_epochs = self._compute_max_epochs()

    def _init_channels(self):
        first_subject = self.subject_ids_all[0]
        raw_names = self.h5[self.dataset_name][first_subject].attrs["channel_names"]
        names = [n.decode("utf-8") if isinstance(n, bytes) else n for n in raw_names]
        self.keep_indices = [i for i, ch in enumerate(names) if ch not in self.omit_channels]
        return [names[i] for i in self.keep_indices]

    def _compute_max_epochs(self):
        max_len = 0
        for subject_id in self.subject_ids_all:
            data, _ = self._load_patient_record(subject_id)
            if data is not None:
                max_len = max(max_len, data.shape[0])
        return max_len

    def _load_patient_record(self, subject_id):
        grp = self.h5[self.dataset_name][subject_id]
        data = grp["data"][:]
        if self.keep_indices:
            data = data[:, self.keep_indices, :]

        labels = grp.attrs["class_labels"]
        if isinstance(labels[0], bytes):
            labels = [l.decode("utf-8") for l in labels]

        for i in reversed(range(3)):
            if i < len(labels) and labels[i] in self.label_map_internal:
                internal_label = int(self.label_map_internal[labels[i]])
                break
        else:
            return None, None, None

        # Apply epoch-level filtering
        if "epoch_mask" in grp:
            epoch_mask = grp["epoch_mask"][:].astype(bool)
            data = data[epoch_mask[:data.shape[0]]]

        return data, internal_label

    def generator(self):
        while True:
            sampled_ids = random.sample(self.subject_ids_all, self.batch_size)
            patient_data = []
            patient_masks = []
            patient_labels = []

            for subject_id in sampled_ids:
                data, label = self._load_patient_record(subject_id)
                if data is None:
                    continue
                patient_data.append(data)
                patient_masks.append(np.ones((data.shape[0],), dtype=bool))
                patient_labels.append(label)

            max_len = self.max_epochs
            padded_data = []
            padded_masks = []

            for data, mask in zip(patient_data, patient_masks):
                pad_width = ((0, max_len - data.shape[0]), (0, 0), (0, 0))
                padded = np.pad(data, pad_width, mode='constant')
                padded_data.append(padded)
                mask_padded = np.pad(mask, (0, max_len - len(mask)), mode='constant', constant_values=False)
                padded_masks.append(mask_padded)

            batch_data = np.stack(padded_data)[..., np.newaxis]  # [B, E, C, T, 1]
            batch_mask = np.stack(padded_masks)                 # [B, E]
            batch_labels = np.array(patient_labels, dtype=np.int32)  # [B]

            yield {
                "data": batch_data.astype(np.float32),
                "epoch_mask": batch_mask,
                "internal_label": batch_labels
            }

    def get_tf_dataset(self):
        output_signature = {
            "data": tf.TensorSpec(
                shape=(self.batch_size, self.max_epochs, len(self.keep_indices), None, 1),
                dtype=tf.float32
            ),
            "epoch_mask": tf.TensorSpec(
                shape=(self.batch_size, self.max_epochs),
                dtype=tf.bool
            ),
            "internal_label": tf.TensorSpec(
                shape=(self.batch_size,),
                dtype=tf.int32
            )
        }
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def close(self):
        if self.h5:
            self.h5.close()
            self.h5 = None

def load_eeg_contrastive_and_patient_datasets(
    h5_file_path,
    dataset_name,
    label_config,
    val_frac=0.2,
    buffer_size_train=32,
    buffer_size_val=16,
    batch_size_buffer=16,
    batch_size_patient=4,
    omit_channels=None,
    seed=42
):
    """
    Splits EEG subjects into train and validation groups and loads both buffer-based and patient-based datasets.

    Returns:
        train_buffer (EEGContrastiveBufferDatasetTF)
        val_buffer (EEGContrastiveBufferDatasetTF)
        train_patient (EEGPatientBatchDatasetTF)
        val_patient (EEGPatientBatchDatasetTF)
    """
    with h5py.File(h5_file_path, "r") as f:
        subject_ids = sorted(list(f[dataset_name].keys()))

    # Shuffle and split
    random.seed(seed)
    random.shuffle(subject_ids)
    n_total = len(subject_ids)
    n_val = round(n_total * val_frac)
    val_ids = subject_ids[:n_val]
    train_ids = subject_ids[n_val:]

    if len(train_ids) == 0 or len(val_ids) == 0:
        raise ValueError("Not enough data for training or validation split. "
                         f"{len(train_ids)=}, {len(val_ids)=}, {n_total=}")

    # Buffered datasets (Phase 1)
    train_buffer = EEGContrastiveBufferDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        buffer_size=buffer_size_train,
        batch_size=batch_size_buffer,
        omit_channels=omit_channels,
        subject_ids=train_ids
    )

    val_buffer = EEGContrastiveBufferDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        buffer_size=buffer_size_val,
        batch_size=batch_size_buffer,
        omit_channels=omit_channels,
        subject_ids=val_ids
    )

    # Patient batch datasets (Phase 2)
    train_patient = EEGPatientBatchDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        batch_size=batch_size_patient,
        omit_channels=omit_channels,
        subject_ids=train_ids
    )

    val_patient = EEGPatientBatchDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        batch_size=batch_size_patient,
        omit_channels=omit_channels,
        subject_ids=val_ids
    )

    return train_buffer, val_buffer, train_patient, val_patient

if __name__ == "__main__":
    # === Path & Dataset Setup ===
    h5_file_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    dataset_name = "combined_south_africa_monopolar_standard_10_20"
    LABEL_CONFIG = {
        "label_map": {
            "neurotypical": 0,
            "generalized": 1,
            "left": 2,
            "right": 3
        },
        "inverse_label_map": {
            0: "neurotypical",
            1: "generalized",
            2: "left",
            3: "right"
        }
    }

    # === Config Parameters ===
    buffer_size_train = 32
    buffer_size_val = 16
    batch_size = 4
    val_frac = 0.2
    omit_channels = None  # or e.g. ["EKG", "StimMarker"]

    # === Load Datasets ===
    train_buffer, val_buffer, train_patient, val_patient = load_eeg_contrastive_and_patient_datasets(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=LABEL_CONFIG,
        val_frac=val_frac,
        buffer_size_train=buffer_size_train,
        buffer_size_val=buffer_size_val,
        batch_size=batch_size,
        omit_channels=omit_channels,
        seed=42
    )

    # === Test Drawing from Buffer Dataset ===
    print("=== Testing Buffer Dataset ===")
    train_ds = train_buffer.get_tf_dataset()
    sample_batch = next(iter(train_ds))
    print("Buffer data shape:", sample_batch["data"].shape)          # [B, C, T, 1]
    print("Buffer labels shape:", sample_batch["internal_label"].shape)  # [B]

    # === Test Drawing from Patient Dataset ===
    print("\n=== Testing Patient Dataset ===")
    patient_ds = train_patient.get_tf_dataset()
    patient_batch = next(iter(patient_ds))
    print("Patient data shape:", patient_batch["data"].shape)          # [B, E, C, T, 1]
    print("Patient mask shape:", patient_batch["epoch_mask"].shape)    # [B, E]
    print("Patient labels shape:", patient_batch["internal_label"].shape)  # [B]