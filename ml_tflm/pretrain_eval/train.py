import tensorflow as tf

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

import numpy as np
import random
import gc

from ml_tflm.pretrain_eval.dataset import build_datasets
from ml_tflm.pretrain_eval.loss import get_cls_loss, get_pre_loss
from ml_tflm.pretrain_eval.model import configure_model, get_classifier
from ml_tflm.pretrain_eval.steps import get_pre_train_step, get_cls_train_step, get_cls_val_step

from ml_tflm.training.trainer import print_weight_updates
from ml_tflm.pre_training.model_pretrain import L2Normalization

def train(supervision, temperature, vicreg_weight,
          F1=8, F2=16, bottleneck_dim=16, proj_dim=64, hidden_dim=128,
          buffer_size_per_class=3, batch_size=16, num_views=2,
          epochs=30, steps_per_epoch=100, val_interval=10, verbose=1, seed=42):
    # Set seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Hardcoded paths
    pre_h5_path = "ml_tflm/dataset/agenda_data_01/augmented_buffered.h5"
    cls_h5_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    cls_h5_name = "combined_south_africa_monopolar_standard_10_20"
    label_json_file = "ml_tflm/training/label_map.JSON"
    aug_json_file = 'ml_tflm/dataset/agenda_data_01/augmented_bin_info.json'

    # Pretraining loss and model setup
    pre_loss = get_pre_loss(supervision, temperature, vicreg_weight)
    model_dict = configure_model(
        feature_args={"dropout_rate": 0.5, "F1": F1, "F2": F2, "bottleneck_dim": bottleneck_dim},
        projector_args={"input_dim": bottleneck_dim, "projection_dim": proj_dim, "hidden_dim": hidden_dim},
    )
    optimizer_dict = {
        "feature_extractor": tf.keras.optimizers.Adam(learning_rate=1e-3),
        "projector": tf.keras.optimizers.Adam(learning_rate=1e-3),
    }

    # Load datasets
    pre_train_ds, cls_train_dataset, cls_val_dataset = build_datasets(
        pre_h5_path=pre_h5_path, num_classes=3,
        label_json_file=label_json_file, cls_h5_path=cls_h5_path, cls_h5_name=cls_h5_name,
        patient_bin_json_path=aug_json_file,
        buffer_size_per_class=buffer_size_per_class, batch_size=batch_size, num_views=num_views,
        verbose=verbose
    )

    pre_train_step = get_pre_train_step(model_dict, optimizer_dict, pre_loss)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        if epoch > 0:
            pre_train_ds._reload_buffer()

        train_loss = 0.0
        train_generator = pre_train_ds.generator()

        for step in range(steps_per_epoch):
            train_batch = next(train_generator)
            loss = pre_train_step(train_batch)
            train_loss += loss.numpy()

        train_loss /= steps_per_epoch

        if verbose >= 1:
            print(f"[Epoch {epoch+1}] Pretrain avg loss: {train_loss:.4f}")
            
        # Run classifier eval every val_interval epochs
        if (epoch + 1) % val_interval == 0 or (epoch + 1) == epochs:
            val_loss = train_cls(model_dict, cls_train_dataset, cls_val_dataset, label_json_file, verbose=verbose)
            if verbose >= 1:
                print(f"[Epoch {epoch+1}] Classifier eval loss: {val_loss:.4f}")
        else:
            val_loss = float("inf")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if verbose >= 1:
                print(f"[Epoch {epoch+1}] New best val loss: {best_val_loss:.4f}")

    return best_val_loss

def train_cls(model_dict, train_dataset, val_dataset, label_json_file,
              epochs=15, steps_per_epoch=240, verbose=1):
    # Before you create a new model and optimizer:
    tf.keras.backend.clear_session()

    cls_loss = get_cls_loss(label_json_file)
    cls_model = get_classifier(model_dict)

    # Dummy input to build variables eagerly (e.g., batch=1, epochs=5, channels=21, time=128)
    dummy_input = tf.zeros([1, 5, 21, 128], dtype=tf.float32)
    _ = cls_model(dummy_input, training=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    best_loss = float("inf")

    cls_train_step = get_cls_train_step(cls_model, optimizer, cls_loss)
    cls_val_step = get_cls_val_step(cls_model, cls_loss)

    for epoch in range(1, epochs + 1):
        # Capture weights before epoch
        old_weights = {
            "pool": [tf.identity(w) for w in cls_model.pool.trainable_variables],
            "classifier": [tf.identity(w) for w in cls_model.classifier.trainable_variables],
        }

        train_loss_accum = 0.0
        train_batches = 0

        for step, batch in enumerate(train_dataset.take(steps_per_epoch)):
            try:
                batch_loss = cls_train_step(batch)
                train_loss_accum += batch_loss.numpy()
                train_batches += 1
            except tf.errors.ResourceExhaustedError:
                if verbose >= 1:
                    print(f"[Epoch {epoch} | Step {step}] Skipping batch due to memory exhaustion.")
                tf.keras.backend.clear_session()
                gc.collect()
                continue

        avg_train_loss = train_loss_accum / train_batches if train_batches > 0 else float('nan')

        if verbose >= 2:
            print(f"[Epoch {epoch}] Avg train loss: {avg_train_loss:.4f}")

        for_val_loss = 0.0
        val_batches = 0

        for step, batch in enumerate(val_dataset):  # Exhaust entire validation set
            try:
                batch_loss = cls_val_step(batch)
                for_val_loss += batch_loss.numpy()
                val_batches += 1
            except tf.errors.ResourceExhaustedError:
                if verbose >= 1:
                    print(f"[Epoch {epoch} | Val Step {step}] Skipping batch due to memory error.")
                tf.keras.backend.clear_session()
                gc.collect()
                continue

        avg_val_loss = for_val_loss / val_batches if val_batches > 0 else float('nan')

        if verbose >= 2:
            print(f"[Epoch {epoch}] Avg val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if verbose >= 1:
                print(f"[Epoch {epoch}] New best val loss: {best_loss:.4f}")

        new_weights = {
            "pool": [tf.identity(w) for w in cls_model.pool.trainable_variables],
            "classifier": [tf.identity(w) for w in cls_model.classifier.trainable_variables],
        }

        if verbose >= 2:
            print_weight_updates(old_weights, new_weights)

    return best_loss

def test_pretrained(idx):
    model_dict = {}
    feature_extractor_path = f'feature_extractor_checkpoints/epoch_{idx}/feature_extractor.keras'
    projector_path = f'feature_extractor_checkpoints/epoch_{idx}/projector.keras'

    model_dict["feature_extractor"] = tf.keras.models.load_model(
        feature_extractor_path,
        custom_objects={"L2Normalization": L2Normalization}
    )

    model_dict["projector"] = tf.keras.models.load_model(
        projector_path,
        custom_objects={"L2Normalization": L2Normalization}
    )

    # Hardcoded paths
    pre_h5_path = "ml_tflm/dataset/agenda_data_01/augmented_buffered.h5"
    cls_h5_path = "ml_tflm/dataset/agenda_data_01/combined_south_africa_monopolar_standard_10_20.h5"
    cls_h5_name = "combined_south_africa_monopolar_standard_10_20"
    label_json_file = "ml_tflm/training/label_map.JSON"
    aug_json_file = 'ml_tflm/dataset/agenda_data_01/augmented_bin_info.json'

    # Load datasets
    _, cls_train_dataset, cls_val_dataset = build_datasets(
        pre_h5_path=pre_h5_path, num_classes=3,
        label_json_file=label_json_file, cls_h5_path=cls_h5_path, cls_h5_name=cls_h5_name,
        patient_bin_json_path=aug_json_file,
        load_pretrain=False
    )

    return train_cls(model_dict, cls_train_dataset, cls_val_dataset, label_json_file, verbose=2)

def run_experiments():
    supervision_options = [True, False]
    feature_extractor_sizes = [
        {"F1": 8, "F2": 16},
        {"F1": 16, "F2": 32}
    ]
    augmentation_settings = [
        {"batch_size": 256, "num_views": 8},
        {"batch_size": 1024, "num_views": 2}
    ]

    experiment_id = 1
    results = []

    for supervision in supervision_options:
        for fe_size in feature_extractor_sizes:
            for aug in augmentation_settings:
                print(f"\n=== Experiment {experiment_id} ===")
                print(f"Supervision: {supervision}")
                print(f"Feature Extractor: F1={fe_size['F1']}, F2={fe_size['F2']}")
                print(f"Augmentation: batch_size={aug['batch_size']}, num_views={aug['num_views']}\n")

                best_loss = train(
                    supervision=supervision,
                    temperature=0.3,
                    vicreg_weight=0.5,
                    F1=fe_size["F1"],
                    F2=fe_size["F2"],
                    bottleneck_dim=16,
                    proj_dim=64,
                    hidden_dim=128,
                    buffer_size_per_class=3,
                    batch_size=aug["batch_size"],
                    num_views=aug["num_views"],
                    epochs=30,
                    steps_per_epoch=100,
                    val_interval=10,
                    verbose=1,
                    seed=42,
                )
                print(f"Experiment {experiment_id} completed. Best val loss: {best_loss:.4f}")
                results.append({
                    "experiment_id": experiment_id,
                    "supervision": supervision,
                    "F1": fe_size["F1"],
                    "F2": fe_size["F2"],
                    "batch_size": aug["batch_size"],
                    "num_views": aug["num_views"],
                    "best_val_loss": best_loss
                })
                experiment_id += 1

    return results

if __name__ == "__main__":
    # all_results = run_experiments()
    idx_list = [30,]

    for idx in idx_list:
        print("="*40)
        print(f"Investigating IDX={idx}...")
        print("="*40)
        print(test_pretrained(idx))

