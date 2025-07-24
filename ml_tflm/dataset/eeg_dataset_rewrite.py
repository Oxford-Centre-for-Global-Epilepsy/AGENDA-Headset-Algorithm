import h5py
import numpy as np
import tensorflow as tf
import atexit

from collections import Counter

class EEGRecordingDatasetTF:
    def __init__(self, h5_file_path, dataset_name, label_config, transform=None, omit_channels=None, subject_ids=None, mirror_flag=False):
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
        self.keep_indices = [i for i, ch in enumerate(self.original_channel_names) if ch not in self.omit_channels]
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

        shape_sample = self.__getitem__(0)
        self.output_signature = {
            "data": tf.TensorSpec(shape=shape_sample["data"].shape, dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(3,), dtype=tf.int32),
            "label_mask": tf.TensorSpec(shape=(3,), dtype=tf.bool),
            "attention_mask": tf.TensorSpec(shape=(self.max_epochs,), dtype=tf.bool),
            "subject_id": tf.TensorSpec(shape=(), dtype=tf.string),
            "internal_label": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

    def __len__(self):
        return len(self.sample_metadata)

    def __getitem__(self, idx): 
        subject = self.sample_metadata[idx]

        subject_id = subject["subject_id"]
        subj_group = self.h5[self.dataset_name][subject_id]
        data = subj_group["data"][()]

        if self.keep_indices:
            data = data[:, self.keep_indices, :]

        if subject["is_mirrored"] and self.mirror_permu is not None:
            data = data[:, self.mirror_permu, :]

        # Load labels from the lookup list
        labels = subject["labels"]
        label_mask = subject["label_mask"]
        internal_label = subject["internal_label"]

        n_epochs, _, n_time = data.shape
        data = data.astype(np.float32)
        if n_epochs < self.max_epochs:
            pad = np.zeros((self.max_epochs - n_epochs, data.shape[1], data.shape[2]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        elif n_epochs > self.max_epochs:
            data = data[:self.max_epochs]

        attention_mask = np.zeros((self.max_epochs,), dtype=bool)
        attention_mask[:min(n_epochs, self.max_epochs)] = True

        return {
            "data": data,
            "labels": labels,
            "label_mask": label_mask,
            "attention_mask": attention_mask,
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
            ds = ds.shuffle(len(self.sample_metadata))
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


if __name__ == "__main__":
    # === Example usage ===

    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }

    dataset = EEGRecordingDatasetTF(
        h5_file_path="ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config, 
        mirror_flag=True
        )

    hist = dataset.get_label_histogram()
    print(hist)


    dataset.close()
