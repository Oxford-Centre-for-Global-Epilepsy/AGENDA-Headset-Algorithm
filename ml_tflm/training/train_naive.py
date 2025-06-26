from ml_tflm.models_tf.classifiers import EEGNetFlatClassifier
from ml_tflm.dataset.eeg_dataset import EEGRecordingTFGenerator
import ml_tflm.training.train_utils as utils
from ml_tflm.training.loss import StructureAwareLoss

import numpy as np
import tensorflow as tf
import os

if __name__ == "__main__":
    

    # Load label configuration
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    # Load EEG datasets
    train_dataset, val_dataset, test_dataset = utils.load_eeg_datasets_split(
        h5_file_path="ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5",
        dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20",
        label_config=label_config
    )
    
    class_hist = utils.compute_label_histogram(train_dataset, label_config)

    print("Class histogram:", class_hist)

    # Create the loss evalutaion function
    loss_fn = StructureAwareLoss(
        label_config=label_config,
        temperature=5.0,
        class_histogram=class_hist,
    )

    # Step 2: Extract input shape info from output_signature
    output_sig = train_dataset.element_spec
    data_spec = output_sig["data"]  # tf.TensorSpec(shape=(E, C, T), ...)

    # Step 3: Use to configure EEGNet
    _, E, C, T = data_spec.shape

    eegnet_args = {
        "num_channels": C,
        "num_samples": T,
        "F1": 4,
        "D": 1,
        "F2": 4,
        "dropout_rate": 0.5,
        "kernel_length": 64,
        "activation": tf.nn.elu
    }

    pooling_args = {
        "hidden_dim": 8,            # or 128 if your model is larger
        "activation": tf.nn.tanh     # or tf.nn.relu, tf.nn.elu depending on your preference
    }

    # Create the classifier model
    model = EEGNetFlatClassifier(
        eegnet_args=eegnet_args,
        pooling_args=pooling_args,
        num_classes=4  # Number of valid hierarchical class combinations
    )

    # === Optimizer ===
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # === Metrics ===
    train_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()

    # === Train Step ===
    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            outputs = model(
                batch["data"],
                attention_mask=batch["attention_mask"],
                training=True
            )
            loss = loss_fn(batch["internal_label"], outputs["logits"])
        grads = tape.gradient(loss, model.trainable_variables)
        
        # Print gradient norm (helps diagnose vanishing grads)
        tf.print("Gradient norm:", tf.linalg.global_norm(grads))
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss_metric.update_state(loss)

    # === Validation Step ===
    @tf.function
    def val_step(batch):
        outputs = model(
            batch["data"],
            attention_mask=batch["attention_mask"],
            training=False
        )
        loss = loss_fn(batch["internal_label"], outputs["logits"])
        val_loss_metric.update_state(loss)

    # === Training Loop ===
    for epoch in range(1, 21):  # 20 epochs
        print(f"\nEpoch {epoch} / 20")

        train_loss_metric.reset_state()
        val_loss_metric.reset_state()

        # Training
        for step, batch in enumerate(train_dataset.take(14)):
            train_step(batch)

        # Validation
        for batch in val_dataset:
            val_step(batch)

        print(f"Epoch {epoch} Summary: Train Loss = {train_loss_metric.result().numpy():.4f}, "
              f"Val Loss = {val_loss_metric.result().numpy():.4f}")

