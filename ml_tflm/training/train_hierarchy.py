from ml_tflm.models_tf.classifiers import EEGNetHierarchicalClassifier
import ml_tflm.training.train_utils as utils
from ml_tflm.training.loss import HierarchicalLoss

import tensorflow as tf

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
    loss_fn = HierarchicalLoss()

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
    model = EEGNetHierarchicalClassifier(
        eegnet_args=eegnet_args,
        pooling_args=pooling_args
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    EPOCHS = 20

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = tf.keras.metrics.Mean()
        
        # --- Training ---
        for batch in train_dataset:
            x = batch["data"]
            y = batch["labels"]
            m = batch["label_mask"]
            attn_mask = batch["attention_mask"]

            with tf.GradientTape() as tape:
                outputs = model(x, training=True, attention_mask=attn_mask)
                loss = loss_fn({"targets": y, "label_mask": m}, outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss.update_state(loss)

        print(f"Train Loss: {train_loss.result():.4f}")

        # --- Validation ---
        val_loss = tf.keras.metrics.Mean()
        for batch in val_dataset:
            x = batch["data"]
            y = batch["labels"]
            m = batch["label_mask"]
            attn_mask = batch["attention_mask"]

            outputs = model(x, training=False, attention_mask=attn_mask)
            loss = loss_fn({"targets": y, "label_mask": m}, outputs)
            val_loss.update_state(loss)

        print(f"Val Loss: {val_loss.result():.4f}")

