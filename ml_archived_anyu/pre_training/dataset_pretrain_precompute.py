import h5py
import numpy as np
import tensorflow as tf
import random
from collections import defaultdict


class EEGAugmentedBufferDatasetTF:
    def __init__(self, h5_path, allowed_bins, num_classes,
                 buffer_size_per_class=3, batch_size=16, num_views=2):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        self.allowed_bins = allowed_bins
        self.num_classes = num_classes
        self.buffer_size_per_class = buffer_size_per_class
        self.batch_size = batch_size
        self.num_views = num_views

        self._gather_bin_map()

        self._load_buffer()

    def _gather_bin_map(self):
        self.bin_map = defaultdict(list)
        for class_key, class_group in self.h5.items():
            if class_key.startswith("class_"):
                class_id = int(class_key.split("class_")[-1])
            else:
                class_id = int(class_key)
            for bin_key in class_group:
                if bin_key in self.allowed_bins:
                    self.bin_map[class_id].append((class_key, bin_key))

    def _load_buffer(self):
        self.buffer_by_class = defaultdict(list)

        for class_id in range(self.num_classes):
            chosen_bins = random.sample(
                self.bin_map[class_id],
                min(self.buffer_size_per_class, len(self.bin_map[class_id]))
            )
            for class_key, bin_id in chosen_bins:
                bin_group = self.h5[class_key][bin_id]
                for segment_id in bin_group:
                    segment_group = bin_group[segment_id]
                    views = segment_group["augmented"][:]
                    self.buffer_by_class[class_id].append(views)

    def _reload_buffer(self):
        self._load_buffer()

    def generator(self):
        while True:
            base_per_class = self.batch_size // self.num_classes
            remainder = self.batch_size % self.num_classes

            per_class_counts = {i: base_per_class for i in range(self.num_classes)}
            per_class_counts[self.num_classes - 1] += remainder

            data = []
            internal_label = []
            augmentation_id = []
            sample_index = []
            vicreg_indices = []

            sample_counter = 0

            for class_id, n_samples in per_class_counts.items():
                segments = self.buffer_by_class[class_id]
                if len(segments) < n_samples:
                    chosen_segments = random.choices(segments, k=n_samples)
                else:
                    chosen_segments = random.sample(segments, n_samples)

                for views in chosen_segments:
                    this_sample_indices = []
                    for _ in range(self.num_views):
                        aug_idx = np.random.randint(views.shape[0])
                        view = views[aug_idx]
                        data.append(view)
                        internal_label.append(class_id)
                        augmentation_id.append(aug_idx)
                        sample_index.append(sample_counter)
                        this_sample_indices.append(len(data) - 1)

                    vicreg_indices.append(this_sample_indices[:2])
                    sample_counter += 1

            C, T = data[0].shape
            batch = {
                "data": np.stack(data, axis=0).astype(np.float32),
                "internal_label": np.array(internal_label, dtype=np.int32),
                "augmentation_id": np.array(augmentation_id, dtype=np.int32),
                "sample_index": np.array(sample_index, dtype=np.int32),
                "vicreg_indices": np.array(vicreg_indices, dtype=np.int32)
            }

            yield batch

    def get_output_signature(self):
        dummy_class, dummy_bin = next(iter(self.bin_map.values()))[0]
        first_segment = next(iter(self.h5[dummy_class][dummy_bin].values()))
        C, T = first_segment["augmented"].shape[1:]

        return {
            "data": tf.TensorSpec(shape=(None, C, T), dtype=tf.float32),
            "internal_label": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "augmentation_id": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "sample_index": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            "vicreg_indices": tf.TensorSpec(shape=(None, 2), dtype=tf.int32),
        }


# --- Utility Functions ---
def split_bins(h5_path, train_ratio=0.75, seed=42):
    with h5py.File(h5_path, 'r') as f:
        all_bins = defaultdict(list)
        for class_key, class_group in f.items():
            if class_key.startswith("class_"):
                class_id = int(class_key.split("class_")[-1])
            else:
                class_id = int(class_key)
            for bin_key in class_group:
                all_bins[class_id].append(bin_key)

    train_bins = defaultdict(list)
    val_bins = defaultdict(list)
    for cls, bin_list in all_bins.items():
        random.Random(seed).shuffle(bin_list)
        split_idx = int(len(bin_list) * train_ratio)
        train_bins[cls] = bin_list[:split_idx]
        val_bins[cls] = bin_list[split_idx:]

    return train_bins, val_bins


def build_datasets(h5_path, num_classes, buffer_size_per_class=3, batch_size=16, num_views=2):
    train_bins, val_bins = split_bins(h5_path)

    train_allowed = [bin_id for bins in train_bins.values() for bin_id in bins]
    val_allowed = [bin_id for bins in val_bins.values() for bin_id in bins]

    train_ds = EEGAugmentedBufferDatasetTF(
        h5_path=h5_path,
        allowed_bins=train_allowed,
        num_classes=num_classes,
        buffer_size_per_class=buffer_size_per_class,
        batch_size=batch_size,
        num_views=num_views
    )

    val_ds = EEGAugmentedBufferDatasetTF(
        h5_path=h5_path,
        allowed_bins=val_allowed,
        num_classes=num_classes,
        buffer_size_per_class=buffer_size_per_class,
        batch_size=batch_size,
        num_views=num_views
    )

    return train_ds, val_ds


# --- Testing Script ---
if __name__ == "__main__":
    h5_path = "ml_tflm/dataset/agenda_data_01/augmented_buffered.h5"
    print("splitting dataset")
    train_dataset_obj, val_dataset_obj = build_datasets(
        h5_path=h5_path,
        num_classes=3,
        buffer_size_per_class=2,
        batch_size=8,
        num_views=2
    )

    print("getting one")

    train_ds = tf.data.Dataset.from_generator(
        train_dataset_obj.generator,
        output_signature=train_dataset_obj.get_output_signature()
    )

    print("gen")

    for batch in train_ds.take(10):
        for key, value in batch.items():
            print(f"{key}: shape = {value.shape}")
