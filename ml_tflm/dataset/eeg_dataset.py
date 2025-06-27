import h5py
import numpy as np
from ml.datasets.eeg_dataset import EEGRecordingDataset
import tensorflow as tf


# TensorFlow-compatible generator wrapper for EEG datasets.
class EEGRecordingTFGenerator():
    def __init__(self, h5_file_path, dataset_name, label_config, transform=None, omit_channels=None, subject_ids=None):
        """
        Args:
            h5_file_path (str): Path to combined HDF5 file.
            dataset_name (str): Root group in HDF5 (e.g. 'my_dataset').
            label_map (dict): Mapping from string labels to integer class IDs.
            omit_channels (list of str): Channel names to omit during loading.
            subject_ids (list of str): Subject ids to include in the dataset.
        """
        self.label_map = label_config["label_map"]
        self.inverse_label_map = label_config["inverse_label_map"]

        # Construct an internal label map for tensor indexing
        self.label_map_internal = {key: i for i, key in enumerate(self.label_map.keys())}


        self.dataset = EEGRecordingDataset(
            h5_file_path=h5_file_path,
            dataset_name=dataset_name,
            label_map=self.label_map,
            transform=transform,
            omit_channels=omit_channels,
            subject_ids=subject_ids
        )

        self.output_signature = self.infer_output_signature()


    def __len__(self):
        return len(self.dataset)

    # Helper methods to initialize the generator
    def get_first_sample(self):
        """
        Returns the first sample from the dataset.
        This is useful for debugging and understanding the data structure.
        """
        return self.dataset[0]

    def infer_output_signature(self):
        """
        Returns a dictionary describing the output signature of the generator.
        Due to lack of knowledge of the exact data types, this method draws a sample from the dataset and infers the types.
        """
        # Draw a sample to infer the output signature
        sample = self.get_first_sample()
        max_epochs = sample["data"].shape[0]
        num_channels = sample["data"].shape[1]
        timepoints = sample["data"].shape[2]

        # Create the output signature based on the sample
        output_signature = {
            "data": tf.TensorSpec(shape=(max_epochs, num_channels, timepoints), dtype=tf.float32),
            "labels": tf.TensorSpec(shape=(3,), dtype=tf.int32),
            "label_mask": tf.TensorSpec(shape=(3,), dtype=tf.bool),
            "attention_mask": tf.TensorSpec(shape=(max_epochs,), dtype=tf.bool),
            "subject_id": tf.TensorSpec(shape=(), dtype=tf.string),
            "internal_label": tf.TensorSpec(shape=(), dtype=tf.int32),
        }

        return output_signature

    def cast_sample(self, sample):
        # Convert standard fields
        out = {
            k: (
                np.bytes_(v.encode("utf-8")) if isinstance(v, str)
                else v.numpy() if hasattr(v, 'numpy') else np.asarray(v)
            )
            for k, v in sample.items()
        }

        # Compute internal label
        label_vec = out["labels"]
        for i in reversed(range(3)):
            if label_vec[i] != -1:
                label_id = label_vec[i]
                break
        else:
            raise ValueError(f"Invalid label vector: all -1")
        
        string_label = self.inverse_label_map[label_id]
        internal_index = self.label_map_internal[string_label]
        out["internal_label"] = np.int32(internal_index)

        return out

    def generator(self):
        for i in range(len(self.dataset)):
            yield self.cast_sample(self.dataset[i])

    def as_dataset(self, batch_size=1, shuffle=False, num_parallel_calls=tf.data.AUTOTUNE):
        dataset = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=self.output_signature
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.dataset))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=num_parallel_calls)
        return dataset

    # The following methods are merely wrappers around the EEGRecordingDataset methods
    # To check the output types, see the EEGRecordingDataset class.
    def get_channel_names(self):
        return self.dataset.get_channel_names()

    def get_omitted_channel_names(self):
        return self.dataset.get_omitted_channel_names()

    def get_num_channels(self):
        return self.dataset.get_num_channels()

    def get_subject_ids(self):
        return self.dataset.get_subject_ids()

    def get_subjects_with_labels(self):
        return self.dataset.get_subjects_with_labels()

    def filter_subjects(self, subject_ids):
        self.dataset.filter_subjects(subject_ids)

    def get_target_labels(self):
        return self.dataset.get_target_labels()

    def get_class_weights(self):
        return self.dataset.get_class_weights()


# The following code is for testing the EEGRecordingTFGenerator class.
# It's not part of the class itself, and not intended for production use.
if __name__ == "__main__":
    # Configure the gererator with the path to the HDF5 file and dataset name
    generator = EEGRecordingTFGenerator(
        h5_file_path="ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5",
        dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20"
    )

    # access the dataset as a TensorFlow dataset
    dataset = generator.as_dataset(batch_size=1, shuffle=True)

    # Iterate through the dataset and print the first few samples to verify the output
    for i, sample in enumerate(dataset):
        print(f"--- Batch {i+1} ---")
        print("Data shape:", sample["data"].shape)
        print("Labels:", sample["labels"].numpy())
        print("Label mask:", sample["label_mask"].numpy())
        print("Attention mask:", sample["attention_mask"].numpy())
        print("Subject ID:", sample["subject_id"].numpy())


        # Only print the first 3 samples for brevity
        if i >= 2:
            break