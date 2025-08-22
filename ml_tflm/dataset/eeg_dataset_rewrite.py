import h5py
import numpy as np
import tensorflow as tf
import atexit

from collections import Counter

class EEGRecordingDatasetTF:
    def __init__(self, h5_file_path, dataset_name, label_config, 
                 transform=None, omit_channels=None, subject_ids=None, mirror_flag=False, 
                 chunk_size=None, deterministic_draw=False):
        # Initialize the dataset access
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.h5 = h5py.File(h5_file_path, "r")
        atexit.register(self.close)

        # Handle the human-readable map and internal label map
        self.label_map = label_config["label_map"]
        self.inverse_label_map = label_config["inverse_label_map"]
        self.label_map_internal = {key: i for i, key in enumerate(self.label_map.keys())}
        self.inverse_label_map_internal = {v: k for k, v in self.label_map_internal.items()}

        self.transform = transform
        self.omit_channels = set(omit_channels or [])

        all_subject_ids = list(self.h5[self.dataset_name].keys())
        self.subject_ids = subject_ids if subject_ids is not None else all_subject_ids
        self.max_epochs = self.h5.attrs["max_epochs"]

        self.build_lookup()

        # Get the channel names
        first_subject = self.subject_ids[0]
        channel_names_raw = self.h5[self.dataset_name][first_subject].attrs["channel_names"]
        self.original_channel_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in channel_names_raw]

        # Normalize channel name for comparison
        def normalize(ch): return ch.upper().replace("EEG ", "").strip()

        normalized_omit = set(normalize(ch) for ch in self.omit_channels)
        self.keep_indices = [i for i, ch in enumerate(self.original_channel_names) if normalize(ch) not in normalized_omit]
        self.channel_names = [self.original_channel_names[i] for i in self.keep_indices]

        if mirror_flag:
            # Extend the lookup table
            self.extend_mirror()

            # Construct the permutation index
            mirror_pairs = [('Fp1', 'Fp2'), ('F3', 'F4'), ('C3', 'C4'), ('P3', 'P4'),
                            ('O1', 'O2'), ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6'), ('A1', 'A2')]
            name_to_idx = {name: i for i, name in enumerate(self.channel_names)}
            mirror_idx = list(range(len(self.channel_names)))
            for left, right in mirror_pairs:
                if left in name_to_idx and right in name_to_idx:
                    i, j = name_to_idx[left], name_to_idx[right]
                    mirror_idx[i], mirror_idx[j] = j, i
            self.mirror_permu = np.array(mirror_idx, dtype=np.int32)
        else:
            self.mirror_permu = None

        self.chunk_size = chunk_size
        # If chunk_size is set, override max_epochs
        if chunk_size is not None:
            self.max_epochs = chunk_size

        # Method for stability in validation
        if deterministic_draw and chunk_size is not None:
            self.precompute_draws(k=chunk_size)
        self.deterministic_draw = deterministic_draw


        shape_sample = self.__getitem__(0)
        self.output_signature = {
            "data": tf.TensorSpec(shape=shape_sample["data"].shape, dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(3,), dtype=tf.int32),
            "label_mask": tf.TensorSpec(shape=(3,), dtype=tf.bool),
            "attention_mask": tf.TensorSpec(shape=(self.max_epochs,), dtype=tf.bool),
            "subject_id": tf.TensorSpec(shape=(), dtype=tf.string),
            "internal_label": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        # Optional VERY HACKY trick
        # self.recast_labels({3:0, 5:3, 4:3})

    def __len__(self):
        return len(self.sample_metadata)

    def __getitem__(self, idx): 
        subject = self.sample_metadata[idx]
        subject_id = subject["subject_id"]
        subj_group = self.h5[self.dataset_name][subject_id]

        full_n_epochs, n_channels, n_time = subj_group["data"].shape
        k = self.chunk_size if self.chunk_size is not None else self.max_epochs

        # === Select contiguous block ===
        if full_n_epochs >= k:
            if self.deterministic_draw and self.chunk_size is not None:
                start = subject["start_index"]
            else:
                start = np.random.randint(0, full_n_epochs - k + 1)

            data = subj_group["data"][start:start + k]
            attention_mask = np.ones((k,), dtype=bool)

        else:
            # Load full data then pad
            data = subj_group["data"][()]
            pad = np.zeros((k - full_n_epochs, n_channels, n_time), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

            attention_mask = np.zeros((k,), dtype=bool)
            attention_mask[:full_n_epochs] = True

        # === Channel selection ===
        if self.keep_indices:
            data = data[:, self.keep_indices, :]

        # === Optional mirroring ===
        if subject["is_mirrored"] and self.mirror_permu is not None:
            data = data[:, self.mirror_permu, :]

        # === Metadata ===
        labels = subject["labels"]
        label_mask = subject["label_mask"]
        internal_label = subject["internal_label"]

        return {
            "data": data.astype(np.float32),               # [k, C, T]
            "labels": labels,
            "label_mask": label_mask,
            "attention_mask": attention_mask,              # [k]
            "subject_id": np.asarray(subject_id.encode("utf-8"), dtype=np.string_),
            "internal_label": np.int32(internal_label)
        }

    def build_lookup(self):
        self.sample_metadata = []

        for sid in self.subject_ids:
            subj_group = self.h5[self.dataset_name][sid]

            labels = subj_group.attrs["class_labels"]
            if isinstance(labels[0], bytes):
                labels = [l.decode("utf-8") for l in labels]

            label_ids = np.full((3,), -1, dtype=np.int32)
            label_mask = np.zeros((3,), dtype=bool)
            for i in range(min(3, len(labels))):
                label_ids[i] = self.label_map.get(labels[i], -1)
                label_mask[i] = label_ids[i] != -1

            for i in reversed(range(3)):
                if label_mask[i]:
                    internal_label = self.label_map_internal.get(labels[i], -1)
                    break
            else:
                continue  # skip if invalid

            self.sample_metadata.append({
                "subject_id": sid,
                "labels": label_ids,
                "label_mask": label_mask,
                "internal_label": internal_label,
                "is_mirrored": False
            })

    def precompute_draws(self, k, seed=42, num_draws=2):
        """
        Precomputes multiple deterministic start indices per subject and replicates sample metadata.

        Args:
            k (int): Number of epochs per segment.
            seed (int): Random seed for reproducibility.
            num_draws (int): Number of segments to draw per subject.
        """
        rng = np.random.default_rng(seed)
        new_metadata = []

        for sample in self.sample_metadata:
            sid = sample["subject_id"]
            full_n_epochs = self.h5[self.dataset_name][sid]["data"].shape[0]

            for draw_idx in range(num_draws):
                if full_n_epochs >= k:
                    start = rng.integers(0, full_n_epochs - k + 1)
                else:
                    start = 0  # Will be padded later

                # Clone the sample and assign the draw
                sample_copy = sample.copy()
                sample_copy["start_index"] = start
                sample_copy["replica_index"] = draw_idx  # optional, for debugging/traceability
                new_metadata.append(sample_copy)

        self.sample_metadata = new_metadata

    def extend_mirror(self):
        new_entries = []

        for subject in self.sample_metadata:
            old_internal_label = subject["internal_label"]
            if old_internal_label == 4:
                new_internal_label = 5
            elif old_internal_label == 5:
                new_internal_label = 4
            else:
                continue

            # Copy label vectors
            new_labels = subject["labels"].copy()
            new_label_mask = subject["label_mask"].copy()

            # Replace the deepest label
            for i in reversed(range(3)):
                if new_label_mask[i]:
                    new_labels[i] = new_internal_label
                    break

            mirrored_subject = {
                "subject_id": subject["subject_id"],
                "labels": new_labels,
                "label_mask": new_label_mask,
                "internal_label": new_internal_label,
                "is_mirrored": True
            }

            new_entries.append(mirrored_subject)

        self.sample_metadata.extend(new_entries)

    def recast_labels(self, new_index):
        """
        Recast internal labels using a new index mapping.

        Args:
            new_index (dict[int → int]): Mapping from old internal label to new internal label.
                                        Subjects with internal labels not in the keys will be dropped.
        """
        updated_metadata = []

        for subject in self.sample_metadata:
            old_label = subject["internal_label"]

            if old_label not in new_index:
                continue  # Drop this subject

            new_label = new_index[old_label]

            try:
                labels, label_mask = make_label_vector_from_internal_label(new_label)
            except ValueError:
                # Invalid new internal label — skip
                continue

            new_subject = subject.copy()
            new_subject["internal_label"] = new_label
            new_subject["labels"] = labels
            new_subject["label_mask"] = label_mask

            updated_metadata.append(new_subject)

        self.sample_metadata = updated_metadata

    def generator(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def as_generator(self, batch_size=1, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=self.output_signature
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.sample_metadata))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=num_parallel_calls)
        return ds
    
    def get_tf_dataset(self, batch_size=1, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
        def tf_getitem(idx):
            def py_getitem(i):
                item = self.__getitem__(int(i))
                return [item[k] for k in self.output_signature]
            out = tf.py_function(py_getitem, [idx], [s.dtype for s in self.output_signature.values()])
            for t, s in zip(out, self.output_signature.values()):
                t.set_shape(s.shape)
            return dict(zip(self.output_signature.keys(), out))

        ds = tf.data.Dataset.range(len(self.sample_metadata))
        if shuffle:
            ds = ds.shuffle(len(self.sample_metadata), reshuffle_each_iteration=True)
        ds = ds.map(tf_getitem, num_parallel_calls=num_parallel_calls)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(1)
        return ds

    def close(self):
        if self.h5:
            self.h5.close()
            self.h5 = None

    def get_channel_names(self):
        return self.channel_names

    def get_omitted_channel_names(self):
        return list(self.omit_channels)

    def get_num_channels(self):
        return len(self.keep_indices)

    def get_subject_ids(self):
        return self.subject_ids

    def get_label_histogram(self):
        counter = Counter()

        for subject in self.sample_metadata:
            internal_label = subject["internal_label"]
            label_str = self.inverse_label_map_internal[internal_label]
            counter[label_str] += 1

        return dict(counter)

def make_label_vector_from_internal_label(internal_label):
    """
    Construct hierarchical labels and mask for a known internal label.

    Args:
        internal_label (int): One of {0, 3, 4, 5}

    Returns:
        labels (np.ndarray): shape [3], int32
        label_mask (np.ndarray): shape [3], bool
    """
    if internal_label == 0:
        labels = np.array([0, -1, -1], dtype=np.int32)
        mask = np.array([True, False, False], dtype=bool)

    elif internal_label == 3:
        labels = np.array([1, 3, -1], dtype=np.int32)
        mask = np.array([True, True, False], dtype=bool)

    elif internal_label == 4:
        labels = np.array([1, 2, 4], dtype=np.int32)
        mask = np.array([True, True, True], dtype=bool)

    elif internal_label == 5:
        labels = np.array([1, 2, 5], dtype=np.int32)
        mask = np.array([True, True, True], dtype=bool)

    else:
        raise ValueError(f"Unsupported internal_label: {internal_label}")

    return labels, mask

if __name__ == "__main__":
    # === Example usage ===

    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }

    dataset = EEGRecordingDatasetTF(
        h5_file_path="ml_tflm/dataset/agenda_data_23_bp45_tr05/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config, 
        mirror_flag=False,
        chunk_size=256, deterministic_draw=False
        )

    hist = dataset.get_label_histogram()
    print(hist)

    # dataset.recast_labels({4: 0, 5:3})
    # hist = dataset.get_label_histogram()
    # print(hist)
    # for idx in range(100):
    #     print(dataset.sample_metadata[idx])

    dataset.close()
