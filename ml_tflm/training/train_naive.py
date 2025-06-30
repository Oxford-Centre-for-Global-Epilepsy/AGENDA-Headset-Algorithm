from ml_tflm.models_tf.classifiers import EEGNetFlatClassifier
import ml_tflm.training.train_utils as utils
from ml_tflm.training.loss import StructureAwareLoss
from ml_tflm.training.cast_prediction import cast_prediction_flat
from ml_tflm.training.metrics import metric_evaluator

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
        label_config=label_config, val_frac=0.1, test_frac=0.1
    )
    
    class_hist = utils.compute_label_histogram(train_dataset, label_config)

    print("Class histogram:", class_hist)

    # Create the loss evalutaion function
    loss_fn = StructureAwareLoss(
        label_config=label_config,
        clip_value=5.0,
        temperature=0.5,
        class_histogram=class_hist,
    )

    # Step 2: Extract input shape info from output_signature
    output_sig = train_dataset.element_spec
    data_spec = output_sig["data"]  # tf.TensorSpec(shape=(E, C, T), ...)

    for sample in train_dataset.take(1):
        print(sample["data"].shape)  # Should print (E, C, T)


    # Step 3: Use to configure EEGNet
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
        "hidden_dim": 64,            # or 128 if your model is larger
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
    evaluator = metric_evaluator(label_config=label_config, prediction_caster=cast_prediction_flat)


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
    EPOCHS = 40

    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch} / {EPOCHS}")

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
        
        # === Metric Evaluation ===
        all_preds = []
        all_targets = []

        for batch in train_dataset:
            x = batch["data"]
            attn_mask = batch["attention_mask"]
            true_labels = batch["internal_label"]

            outputs = model(x, training=False, attention_mask=attn_mask)
            logits = outputs["logits"]  # shape [B, C]

            all_preds.append(logits)
            all_targets.extend(true_labels.numpy().tolist())

        # Stack logits into a single [N, C] tensor
        all_preds_tensor = tf.concat(all_preds, axis=0)

        # Optional: convert targets to tensor
        all_targets_tensor = tf.convert_to_tensor(all_targets, dtype=tf.int32)

        # Now pass to evaluator
        metrics = evaluator.evaluate(all_preds_tensor, all_targets_tensor)
        print("Val F1: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}".format(
            metrics["f1"], metrics["accuracy"], metrics["precision"], metrics["recall"]
        ))
