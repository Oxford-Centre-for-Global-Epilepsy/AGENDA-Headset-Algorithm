import tensorflow as tf
from ml_tflm.pre_training.pretrain_utils import (
    split_augmented_eeg_datasets,
    configure_model,
    train_grouped_supcon
)
from ml_tflm.pre_training.loss_pretrain import InstanceSupConLoss, LabelSupConLoss
from ml_tflm.training.train_utils import load_label_config
import numpy as np
from ml_tflm.pre_training.dataset_pretrain_precompute import build_datasets

def main():
    # === Hyperparameters ===
    batch_size = 1024

    temperature = 0.5
    steps_per_epoch = 100
    val_steps = 50
    total_epochs = 50
    F1 = 16
    F2 = 32
    bottleneck_dim = 16
    proj_dim = 16
    hidden_dim = 32

    num_views = 2
    buffer_size = 2

    # === Construct buffers ===
    print("===== Preparing Datasets =====")
    h5_file_path = "ml_tflm/dataset/agenda_data_01/augmented_buffered.h5"
    train_buffer, val_buffer = build_datasets(
        h5_path=h5_file_path,
        num_classes = 3,
        buffer_size_per_class = buffer_size,
        batch_size=batch_size,
        num_views=num_views
    )

    # === Models and Loss ===
    print("===== Preparing Models =====")
    model_dict = configure_model(
        feature_args={"dropout_rate": 0.5, "F1": F1, "F2": F2, "bottleneck_dim": bottleneck_dim},
        projector_args={"input_dim": bottleneck_dim, "projection_dim": proj_dim, "hidden_dim": hidden_dim},
    )

    # Class-level: weak positives (same-label) + repel same-class negatives
    print("===== Preparing Loss =====")
    loss_fn_class = LabelSupConLoss(temperature=temperature, vicreg_weight=1.0)

    optimizer_dict = {
        "feature_extractor": tf.keras.optimizers.Adam(learning_rate=1e-3),
        "projector": tf.keras.optimizers.Adam(learning_rate=1e-3),
    }

    # === Train ===
    print("===== Train Started =====")
    best_val_loss = train_grouped_supcon(
        model_dict=model_dict,
        optimizer_dict=optimizer_dict,
        loss_fn=loss_fn_class,
        train_buffer=train_buffer,
        val_buffer=val_buffer,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps,
        smoothing_window=3,
    )

    print("Training complete. Best smoothed validation loss:", best_val_loss)

if __name__ == "__main__":
    main()
