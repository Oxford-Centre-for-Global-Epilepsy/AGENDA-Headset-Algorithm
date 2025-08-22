import numpy as np
from scipy.signal import butter, filtfilt
from scipy.special import softmax
import tensorflow as tf
from ml_tflm.pre_training.dataset_pretrain import EEGContrastiveBufferDatasetTF
import matplotlib.pyplot as plt
import random


import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom
from scipy.special import softmax

class EEGBatchAugmentor:
    def __init__(self, logits, fs=128):
        self.fs = fs
        self.logits = np.array(logits)
        self.probs = softmax(self.logits)
        self.augment_methods = [
            self._channel_masking,
            self._time_masking,
            self._channel_jittering,
            self._bandstop_dropout,
            self._gaussian_superpose,
            self._time_scaling
        ]

    def apply(self, batch, labels, num_views=2):
        B, C, T = batch.shape
        data = []
        augment_id = []
        internal_label = []
        sample_index = []
        vicreg_indices = []

        for i in range(B):
            x = batch[i]
            label = labels[i]
            this_sample_indices = []

            for v in range(num_views):
                aug_id = np.random.choice(len(self.augment_methods), p=self.probs)
                x_aug = self.augment_methods[aug_id](x.copy(), label_name=label)
                data.append(x_aug)
                augment_id.append(aug_id)
                internal_label.append(label)
                sample_index.append(i)
                this_sample_indices.append(len(data) - 1)

            vicreg_indices.append(this_sample_indices[:2])

        return {
            "data": np.stack(data, axis=0),
            "internal_label": np.array(internal_label),
            "augment_id": np.array(augment_id),
            "sample_index": np.array(sample_index),
            "vicreg_indices": np.array(vicreg_indices, dtype=np.int32)
        }

    def _channel_masking(self, x, label_name=None):
        num_mask = np.random.randint(1, 4)
        C = x.shape[0]
        mask_indices = np.random.choice(C, size=min(num_mask, C), replace=False)
        x[mask_indices, :] = 0
        return x

    def _time_masking(self, x, label_name=None):
        T = x.shape[1]
        mask_width = np.random.randint(16, 64)
        if mask_width >= T:
            return x
        start = np.random.randint(0, T - mask_width)
        x[:, start:start + mask_width] = 0
        return x

    def _channel_jittering(self, x, label_name=None):
        num_jitter = np.random.randint(1, 3)
        C = x.shape[0]
        jitter_indices = np.random.choice(C, size=min(num_jitter, C), replace=False)
        for idx in jitter_indices:
            x[idx, :] = self._generate_band_limited_noise(x.shape[1])
        return x

    def _bandstop_dropout(self, x, label_name=None):
        f_center = np.random.choice([20.0, 30.0, 50.0])
        bandwidth = np.random.uniform(1.0, 4.0)
        return self._bandstop_filter(x, f_center=f_center, bandwidth=bandwidth)

    def _gaussian_superpose(self, x, label_name=None):
        std = 0.05  # Use a fixed std or tuned globally, not label-dependent
        noise = np.random.randn(*x.shape) * std
        x_noisy = x + noise
        return self._bandpass_filter(x_noisy)

    def _time_scaling(self, x, label_name=None):
        C, T = x.shape
        scale = np.random.uniform(0.8, 1.2)  # fixed range for all labels
        T_new = max(int(T * scale), 1)
        x_scaled = np.zeros_like(x)
        for i in range(C):
            resampled = zoom(x[i], T_new / T)
            if len(resampled) < T:
                pad = T - len(resampled)
                resampled = np.pad(resampled, (0, pad), mode='reflect')
            else:
                resampled = resampled[:T]
            x_scaled[i] = resampled
        return self._bandpass_filter(x_scaled)

    def _generate_band_limited_noise(self, num_samples, f_low=0.5, f_high=40.0, order=4):
        nyq = self.fs / 2
        b, a = butter(order, [f_low / nyq, f_high / nyq], btype='band')
        noise = np.random.randn(num_samples)
        filtered = filtfilt(b, a, noise)
        return (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-6)

    def _bandpass_filter(self, x, f_low=0.5, f_high=40.0, order=4):
        nyq = self.fs / 2
        b, a = butter(order, [f_low / nyq, f_high / nyq], btype='band')
        return filtfilt(b, a, x, axis=-1)

    def _bandstop_filter(self, x, f_center=30.0, bandwidth=2.0, order=4):
        nyq = self.fs / 2
        f_low = max(f_center - bandwidth / 2, 0.1)
        f_high = f_center + bandwidth / 2
        b, a = butter(order, [f_low / nyq, f_high / nyq], btype='bandstop')
        return filtfilt(b, a, x, axis=-1)

class EEGAugmentedBufferDatasetTF(EEGContrastiveBufferDatasetTF):
    def __init__(self, *args, augmentor, num_views=2, **kwargs):
        """
        Args:
            augmentor: An instance of EEGBatchAugmentor with .apply(batch, num_views=N)
            num_views: Number of augmented views per input sample (default 2)
        """
        self.buffer_by_class = {}
        self.min_class_ratio = 0.2

        super().__init__(*args, **kwargs)

        self.augmentor = augmentor
        self.num_views = num_views
        if self.augmentor is None:
            raise ValueError("You must pass an augmentor instance.")
        
        for k, v in self.class_subject_map.items():
            print(f"{k}: {len(v)}")
    
    def _reload_buffer(self):
        # Precompute how many subjects per class
        subjects_per_class = self.buffer_size // self.num_classes
        remainder = self.buffer_size - subjects_per_class * self.num_classes

        selected_subjects = []
        for c in range(self.num_classes):
            class_pool = self.class_subject_map[c]
            n_pick = subjects_per_class + (1 if c == self.num_classes - 1 else 0 and remainder > 0)
            picked = random.sample(class_pool, min(n_pick, len(class_pool)))
            selected_subjects.extend((sid, c) for sid in picked)

        random.shuffle(selected_subjects)  # Optional, for variation across epochs

        buffer_by_class = {c: [] for c in range(self.num_classes)}

        for subject_id, class_id in selected_subjects:
            grp = self.h5[self.dataset_name][subject_id]
            data = grp["data"][()]
            if self.keep_indices:
                data = data[:, self.keep_indices, :]

            mask = np.ones((data.shape[0],), dtype=bool)
            if "epoch_mask" in grp:
                mask = grp["epoch_mask"][()].astype(bool)
                mask = mask[:data.shape[0]]

            for i in range(data.shape[0]):
                if mask[i]:
                    buffer_by_class[class_id].append(data[i])

        self.buffer_by_class = {
            c: np.stack(v) if len(v) > 0 else np.empty((0, data.shape[1], data.shape[2]))
            for c, v in buffer_by_class.items()
        }

    def generator(self):
        base_per_class = self.batch_size // self.num_classes
        remainder = self.batch_size % self.num_classes
        C, T = next(iter(self.buffer_by_class.values())).shape[1:]  # Get (C, T)

        while True:
            # Precompute per-class counts
            per_class_counts = {i: base_per_class for i in range(self.num_classes)}
            per_class_counts[self.num_classes - 1] += remainder  # Assign leftover to last class

            batch_data = []
            batch_labels = []

            for class_id, count in per_class_counts.items():
                class_pool = self.buffer_by_class[class_id]
                pool_size = class_pool.shape[0]

                if pool_size < count:
                    idxs = np.random.choice(pool_size, count, replace=True)
                else:
                    idxs = np.random.choice(pool_size, count, replace=False)

                batch_data.append(class_pool[idxs])
                batch_labels.append(np.full(count, class_id, dtype=np.int32))

            batch_data = np.concatenate(batch_data, axis=0).astype(np.float32)
            batch_labels = np.concatenate(batch_labels, axis=0)

            # Apply augmentations and get VICReg pairing info
            augmented = self.augmentor.apply(batch_data, batch_labels, num_views=self.num_views)

            # Add channel dimension to data
            data = augmented["data"].astype(np.float32)[..., np.newaxis]

            yield {
                "data": data,
                "internal_label": augmented["internal_label"],
                "augmentation_id": augmented["augment_id"],
                "sample_index": augmented["sample_index"],
                "vicreg_indices": augmented["vicreg_indices"]
            }

    def get_tf_dataset(self):
        output_signature = {
            "data": tf.TensorSpec(
                shape=(self.batch_size * self.num_views, len(self.keep_indices), None, 1),
                dtype=tf.float32
            ),
            "internal_label": tf.TensorSpec(
                shape=(self.batch_size * self.num_views,),
                dtype=tf.int32
            ),
            "augmentation_id": tf.TensorSpec(
                shape=(self.batch_size * self.num_views,),
                dtype=tf.int32
            ),
            "sample_index": tf.TensorSpec(
                shape=(self.batch_size * self.num_views,),
                dtype=tf.int32
            )
        }
        ds = tf.data.Dataset.from_generator(self.generator, output_signature=output_signature)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

def build_augmented_dataset(h5_file_path,
                            dataset_name,
                            label_config,
                            augment_logits,
                            batch_size=16,
                            buffer_size=32,
                            num_views=2,
                            omit_channels=None,
                            subject_ids=None):
    """
    Build a TensorFlow dataset with EEG contrastive augmentation.

    Returns:
        dataset_obj: EEGAugmentedBufferDatasetTF instance (with .refresh_buffer())
        tf_dataset: tf.data.Dataset
    """
    # Build the augmentor
    augmentor = EEGBatchAugmentor(
        logits=augment_logits
    )

    # Build the dataset
    dataset = EEGAugmentedBufferDatasetTF(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        batch_size=batch_size,
        buffer_size=buffer_size,
        omit_channels=omit_channels,
        subject_ids=subject_ids,
        augmentor=augmentor,
        num_views=num_views
    )

    return dataset

if __name__ == "__main__": 
    # --- Config ---
    h5_file_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    dataset_name = "combined_south_africa_monopolar_standard_10_20"
    label_config = {
        "label_map": {"neurotypical": 0, "generalized": 1, "left": 2, "right": 3},
        "inverse_label_map": {0: "neurotypical", 1: "generalized", 2: "left", 3: "right"}
    }
    augment_logits = [1.0, 1.0, 2.0, -10, -10, -10]  # Adjust based on available augmentations
    batch_size = 16
    num_views = 4
    buffer_size = 8

    # --- Build dataset ---
    dataset = build_augmented_dataset(
        h5_file_path=h5_file_path,
        dataset_name=dataset_name,
        label_config=label_config,
        augment_logits=augment_logits,
        batch_size=batch_size,
        buffer_size=buffer_size,
        num_views=num_views
    )

    # --- Test generator ---
    dataset._reload_buffer()
    gen = dataset.generator()

    batch = next(gen)

    print("Data shape:", batch["data"].shape)  # Should be (batch_size * num_views, C, T, 1)
    print("Label shape:", batch["internal_label"].shape)
    print("Labels:", batch["internal_label"])
    print("Samples:", batch["sample_index"])
    print("Samples:", batch["vicreg_indices"])

    if "augmentation_id" in batch:
        print("Augment IDs:", batch["augmentation_id"])

    # --- Plot all channels from first augmented sample ---
    sample = batch["data"][0].squeeze()  # shape (C, T)
    label = batch["internal_label"][0]

    plt.figure(figsize=(14, 10))
    offset = 10  # Vertical offset between channels

    for i in range(sample.shape[0]):
        plt.plot(sample[i] + i * offset, label=f"Ch {i}")

    plt.title(f"Augmented Sample 0 | Label = {label}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude (offset per channel)")
    plt.yticks([])
    plt.legend(loc='upper right', ncol=2, fontsize="small")
    plt.tight_layout()
    plt.show()
