import tensorflow as tf
import numpy as np
import json
import os
from tqdm import tqdm
import warnings

from ml_tflm.training.train_utils import load_eeg_datasets_split, load_label_config


def serialize_example(data, label, mask, attention_mask, subject_id, internal_label):
    """
    Serializes a single EEG sample into a tf.train.Example.
    All list fields must be flat and of fixed shape for compatibility with downstream parsing.
    """
    feature = {
        "data": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy()])
        ),
        "labels": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        "label_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=mask)),
        "attention_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=attention_mask)),
        "subject_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[subject_id.encode("utf-8")])),
        "internal_label": tf.train.Feature(int64_list=tf.train.Int64List(value=[internal_label])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def convert_to_flat_int_list(array_like, field_name):
    arr = np.asarray(array_like).flatten()
    if arr.ndim != 1:
        raise ValueError(f"{field_name} must be 1D, got shape {arr.shape}")
    return [int(x) for x in arr]


def extract_scalar_int(value, field_name):
    if isinstance(value, (int, np.integer)):
        return int(value)
    elif np.ndim(value) == 0:
        return int(value.item())
    elif np.ndim(value) == 1 and len(value) == 1:
        return int(value[0])
    else:
        raise ValueError(f"{field_name} must be a scalar or 1-element array, got shape {np.shape(value)}")


def extract_and_write_metadata(first_sample, output_dir):
    shape = first_sample["data"].shape  # (1, epochs, channels, time)
    if len(shape) != 4:
        raise ValueError(f"Expected batched data shape (1, epochs, channels, time), got {shape}")
    meta = {
        "max_epochs": int(shape[1]),
        "num_channels": int(shape[2]),
        "num_time": int(shape[3])
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata written to: {meta_path}")


def write_tf_record_from_tf_dataset(tf_dataset, output_path, write_meta=False):
    """
    Serializes a TensorFlow dataset into a TFRecord file and optionally writes metadata.
    Assumes batch_size=1 for compatibility.
    """
    count = 0
    output_dir = os.path.dirname(output_path)
    first_sample = None

    with tf.io.TFRecordWriter(output_path) as writer:
        for sample in tqdm(tf_dataset, desc=f"Writing {output_path}", unit="sample"):
            if first_sample is None:
                first_sample = sample
                if write_meta:
                    extract_and_write_metadata(first_sample, output_dir)

            try:
                serialized = serialize_example(
                    data=sample["data"].numpy(),
                    label=convert_to_flat_int_list(sample["labels"].numpy(), "labels"),
                    mask=convert_to_flat_int_list(sample["label_mask"].numpy(), "label_mask"),
                    attention_mask=convert_to_flat_int_list(sample["attention_mask"].numpy(), "attention_mask"),
                    subject_id=sample["subject_id"].numpy().item().decode("utf-8"),
                    internal_label=extract_scalar_int(sample["internal_label"].numpy(), "internal_label")
                )
                writer.write(serialized)
                count += 1
            except Exception as e:
                warnings.warn(f"Skipping sample due to error: {e}")

    print(f"Saved {count} samples to {output_path}")


if __name__ == "__main__":
    cls_h5_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    cls_h5_name = "combined_south_africa_monopolar_standard_10_20"
    label_json_file = "ml_tflm/training/label_map.JSON"

    train_val_sets, test_dataset = load_eeg_datasets_split(
        h5_file_path=cls_h5_path,
        dataset_name=cls_h5_name,
        label_config=load_label_config(label_json_file),
        val_frac=0.2,
        test_frac=0.0,
        k_fold=False,
        batch_size=1,  # Important: must be 1 for serialization
        shuffle=False
    )

    train_ds, val_ds = train_val_sets[0]

    write_tf_record_from_tf_dataset(train_ds, "ml_tflm/dataset/agenda_data_01/train.tfrecord", write_meta=True)
    write_tf_record_from_tf_dataset(val_ds, "ml_tflm/dataset/agenda_data_01/val.tfrecord", write_meta=False)
