import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = warning+, 2 = error+, 3 = fatal

from ml_tflm.models_tf.classifiers import EEGNetFlatClassifier
import ml_tflm.training.train_utils as utils
from ml_tflm.training.loss import StructureAwareLoss
from ml_tflm.training.metrics import metric_evaluator
from ml_tflm.training.trainer import Trainer

import tensorflow as tf


def prepare_training_components():
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    train_val_sets, test_dataset = utils.load_eeg_datasets_split(
        h5_file_path="ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5",
        dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20",
        label_config=label_config,
        val_frac=0.3,
        test_frac=0.0
    )

    train_dataset = train_val_sets[0][0]
    val_dataset = train_val_sets[0][1]

    class_hist = utils.compute_label_histogram(train_dataset, label_config)

    loss_fn = StructureAwareLoss(
        label_config=label_config,
        clip_value=5.0,
        temperature=0.5,
        class_histogram=class_hist,
    )

    data_spec = train_dataset.element_spec["data"]
    _, E, C, T = data_spec.shape

    eegnet_args = {
        "num_channels": C,
        "num_samples": T,
        "F1": 8,
        "D": 2,
        "F2": 16,
        "dropout_rate": 0.25,
        "kernel_length": 64,
        "activation": tf.nn.elu
    }

    pooling_args = {
        "hidden_dim": 64,
        "activation": tf.nn.tanh
    }

    model = EEGNetFlatClassifier(
        eegnet_args=eegnet_args,
        pooling_args=pooling_args,
        num_classes=4
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    evaluator = metric_evaluator(
        label_config=label_config,
        prediction_caster="cast_prediction_flat"
    )

    input_keys = {
        "x": "data",
        "attention_mask": "attention_mask",
    }

    target_keys = {
        "targets": "internal_label",
    }

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        evaluator=evaluator,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_input_lookup=input_keys,
        model_target_lookup=target_keys
    )


def main():
    trainer = prepare_training_components()
    trainer.train_loop(epochs=40, steps_per_epoch=10)


if __name__ == "__main__":
    main()
