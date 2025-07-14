import h5py
import numpy as np
import tensorflow as tf
import atexit

class EEGRecordingDatasetTF:
    def __init__(self, h5_file_path, dataset_name, label_config, transform=None, omit_channels=None, subject_ids=None):
        self.h5_file_path = h5_file_path
        self.dataset_name = dataset_name
        self.label_map = label_config["label_map"]
        self.inverse_label_map = label_config["inverse_label_map"]

        # Construct internal label map from keys in label_map (unsorted)
        self.label_map_internal = {key: i for i, key in enumerate(self.label_map.keys())}

        self.transform = transform
        self.omit_channels = set(omit_channels or [])

        self.h5 = h5py.File(h5_file_path, "r")
        atexit.register(self.close)

        all_subject_ids = list(self.h5[self.dataset_name].keys())
        self.subject_ids = subject_ids if subject_ids is not None else all_subject_ids
        self.max_epochs = self.h5.attrs["max_epochs"]

        first_subject = self.subject_ids[0]
        channel_names_raw = self.h5[self.dataset_name][first_subject].attrs["channel_names"]
        self.original_channel_names = [name.decode("utf-8") if isinstance(name, bytes) else name for name in channel_names_raw]
        self.keep_indices = [i for i, ch in enumerate(self.original_channel_names) if ch not in self.omit_channels]
        self.channel_names = [self.original_channel_names[i] for i in self.keep_indices]

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
        return len(self.subject_ids)

    def __getitem__(self, idx): 
        subject_id = self.subject_ids[idx]
        subj_group = self.h5[self.dataset_name][subject_id]
        data = subj_group["data"][()]

        if self.keep_indices:
            data = data[:, self.keep_indices, :]

        labels = subj_group.attrs["class_labels"]
        if isinstance(labels[0], bytes):
            labels = [l.decode("utf-8") for l in labels]

        label_ids = np.full((3,), -1, dtype=np.int32)
        label_mask = np.zeros((3,), dtype=bool)
        for i in range(min(3, len(labels))):
            label_ids[i] = self.label_map.get(labels[i], -1)
            label_mask[i] = label_ids[i] != -1

        # Infer internal label from deepest known label
        for i in reversed(range(3)):
            if label_mask[i]:
                internal_label = self.label_map_internal.get(labels[i], -1)
                break
        else:
            raise ValueError("Invalid label vector: all -1")

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
            "labels": label_ids,
            "label_mask": label_mask,
            "attention_mask": attention_mask,
            "subject_id": np.asarray(subject_id.encode("utf-8"), dtype=np.string_),
            "internal_label": np.int32(internal_label)
        }

    def generator(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def as_generator(self, batch_size=1, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=self.output_signature
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=len(self.subject_ids))
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

        ds = tf.data.Dataset.range(len(self.subject_ids))
        if shuffle:
            ds = ds.shuffle(len(self.subject_ids))
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

if __name__ == "__main__":
    # === Example usage ===

    label_config = {
        "label_map": {"neurotypical": 0, "epileptic": 1, "focal": 2, "generalized": 3, "left": 4, "right": 5},
        "inverse_label_map": {0: "neurotypical", 1: "epileptic", 2: "focal", 3: "generalized", 4: "left", 5: "right"},
    }

    dataset = EEGRecordingDatasetTF(
        h5_file_path="ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config
        )

    tf_dataset = dataset.as_generator(batch_size=4, shuffle=False)

    for batch in tf_dataset.take(1):
        print("=== Batch Contents ===")
        print("data:", batch["data"].shape)  # Expected: [4, max_epochs, channels, time]
        print("attention_mask:", batch["attention_mask"].shape)  # Expected: [4, max_epochs]
        print("internal_label:", batch["internal_label"])  # Expected: [4]
        print("subject_id:", batch["subject_id"])  # Shape: [4]

    dataset.close()
