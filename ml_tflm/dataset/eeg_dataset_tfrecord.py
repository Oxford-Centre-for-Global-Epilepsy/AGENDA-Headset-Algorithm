import tensorflow as tf
import json
import os


def parse_eeg_example(example_proto, max_epochs, num_channels, num_time):
    feature_description = {
        "data": tf.io.FixedLenFeature([], tf.string),
        "labels": tf.io.FixedLenFeature([3], tf.int64),
        "label_mask": tf.io.FixedLenFeature([3], tf.int64),
        "attention_mask": tf.io.FixedLenFeature([max_epochs], tf.int64),
        "subject_id": tf.io.FixedLenFeature([], tf.string),
        "internal_label": tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(example_proto, feature_description)

    data = tf.io.parse_tensor(parsed["data"], out_type=tf.float32)
    data = tf.reshape(data, [max_epochs, num_channels, num_time])

    return {
        "data": data,
        "labels": tf.cast(parsed["labels"], tf.int32),
        "label_mask": tf.cast(parsed["label_mask"], tf.bool),
        "attention_mask": tf.cast(parsed["attention_mask"], tf.bool),
        "subject_id": parsed["subject_id"],
        "internal_label": tf.cast(parsed["internal_label"], tf.int32),
    }


def load_eeg_tfrecord_dataset(tfrecord_path, max_epochs, num_channels, num_time,
                               batch_size=4, shuffle=True, shuffle_buffer_size=256):
    raw_ds = tf.data.TFRecordDataset(tfrecord_path, num_parallel_reads=tf.data.AUTOTUNE)
    parsed_ds = raw_ds.map(
        lambda x: parse_eeg_example(x, max_epochs, num_channels, num_time),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if shuffle:
        parsed_ds = parsed_ds.shuffle(shuffle_buffer_size)
    parsed_ds = parsed_ds.cache()
    parsed_ds = parsed_ds.batch(batch_size)
    parsed_ds = parsed_ds.prefetch(tf.data.AUTOTUNE)
    return parsed_ds


def load_dataset_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)


def infer_dataset_shape(tfrecord_path):
    raw_ds = tf.data.TFRecordDataset(tfrecord_path)
    for raw_example in raw_ds.take(1):
        example = tf.io.parse_single_example(raw_example, {
            "data": tf.io.FixedLenFeature([], tf.string),
        })
        data = tf.io.parse_tensor(example["data"], out_type=tf.float32)
        shape = tf.shape(data)
        return {
            "max_epochs": int(shape[0].numpy()),
            "num_channels": int(shape[1].numpy()),
            "num_time": int(shape[2].numpy()),
        }
    raise ValueError("TFRecord file is empty or unreadable.")


def load_dataset_meta_or_infer(tfrecord_path, meta_path=None):
    if meta_path and os.path.exists(meta_path):
        print(f"Using metadata file: {meta_path}")
        return load_dataset_meta(meta_path)
    else:
        print(f"Inferring dataset shape from: {tfrecord_path}")
        return infer_dataset_shape(tfrecord_path)


if __name__ == "__main__":
    train_path = "ml_tflm/dataset/agenda_data_01/train.tfrecord"
    val_path = "ml_tflm/dataset/agenda_data_01/val.tfrecord"
    meta_path = "ml_tflm/dataset/agenda_data_01/meta.json"  # optional

    meta = load_dataset_meta_or_infer(train_path, meta_path)

    train_ds = load_eeg_tfrecord_dataset(train_path, batch_size=4, shuffle=True, **meta)
    val_ds = load_eeg_tfrecord_dataset(val_path, batch_size=4, shuffle=False, **meta)

    for batch in train_ds.take(100):
        print("Train batch:")
        print("  data shape:", batch["data"].shape)
        print("  attention_mask shape:", batch["attention_mask"].shape)
        print("  internal_label:", batch["internal_label"])
        print("  subject_id:", batch["subject_id"])
