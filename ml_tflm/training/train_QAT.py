import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_LAYOUT_OPTIMIZER"] = "0"
os.environ["TF_AUTOGRAPH_CACHE_DIR"] = "D:/tf_autograph"

# Tensorflow Imports
import tensorflow as tf
tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

# Custom Imports
from ml_tflm.training.loss import BinaryLoss
from ml_tflm.training.trainer import Trainer
import ml_tflm.training.train_utils as utils
from ml_tflm.training.metrics import metric_evaluator
from ml_tflm.models_tf.classifier_QAT import EEGNetFlatClassifierQAT

import json
from pathlib import Path
import numpy as np
import shutil

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float, tf.Tensor)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    # ===== LOADING DATA =====

    # --- Load label config ---
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    # --- define internal label cap ---
    internal_label_cap = {
        0: 450,
        3: 260,
        4: 94,
        5: 96
    }

    # --- load dataset using utils ---
    train_val_sets, test_dataset, label_histograms = utils.prepare_eeg_datasets(
        h5_file_path="ml_tflm/dataset/agenda_data_23_bp45_tr05/merged_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config,
        omit_channels=["A1","A2", "Fz", "Pz", "Cz"],
        val_frac=0.2,
        test_frac=0.15,
        k_fold=True,
        stratify=True,
        internal_label_cap=internal_label_cap,
        batch_size=32,
        mirror_flag=False,
        chunk_size=256
    )
    
    print(" -> Dataset Loaded")
    print("     -> Loaded Dataset Contains:" + str(label_histograms))

    # --- Define metric holder ---
    train_metrics = []
    train_history = {}

    for fold_idx in range(len(train_val_sets)):
        train_val_set = train_val_sets[fold_idx]
        train_dataset = train_val_set[0]
        val_dataset = train_val_set[1]
        
        # --- instantiate the loss functions here ---
        loss_fn = BinaryLoss()
        entropy_loss = None

        # --- get EEGNet shape info ---
        data_spec = train_dataset.element_spec["data"]
        _, E, C, T = data_spec.shape

        print(E, C, T)

        # --- Resolve model args ---
        eegnet_args = {
            "num_channels": C,
            "num_samples": T,
            "F1": 16,
            "D": 2,
            "F2": 4,
            "dropout_rate": 0.5,
            "kernel_length": 64,   
        }
        head_args = {
            "num_classes": 2,
            "l2_weight": 0.0
        }

        model = EEGNetFlatClassifierQAT(eegnet_args, head_args)

        # --- create optimizer ---
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

        # --- create evaluator ---
        evaluator = metric_evaluator(
            label_config=label_config, 
            prediction_caster="cast_prediction_binary"
        )

        # --- create lookup tabel ---
        model_input_lookup = {
            "x": "data",
            "attention_mask": "attention_mask"
        }
        model_target_lookup = {
            "targets": "labels",
            "entropy_targets": "internal_label"
        }

        # --- Trainer ---
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            attention_loss=entropy_loss,
            evaluator=evaluator,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            model_input_lookup=model_input_lookup,
            model_target_lookup=model_target_lookup,
            save_ckpt=True,
            ckpt_interval=1,
            ckpt_save_dir="ml_tflm/training/checkpoints",
            load_ckpt=False,
            ckpt_load_dir="ml_tflm/training/checkpoints",
            attention_warmup_epoch=0,
            anneal_interval=0,
            anneal_coeff=1.0
        )

        print(" -> Training Time!")


        # --- Train ---
        trainer.train_loop(
            epochs=40,
            steps_per_epoch=100
        )

        train_metrics.append(trainer.get_metrics())
        train_history[fold_idx] = trainer.get_metrics(mode="all")

        # === Copy the whole checkpoint folder for this fold ===
        src_dir = "ml_tflm/training/checkpoints"
        dst_dir = f"ml_tflm/training/checkpoints_fold{fold_idx}"
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        # === Clear the source directory completely ===
        shutil.rmtree(src_dir)        # remove everything
        os.makedirs(src_dir, exist_ok=True)  # recreate empty dir

    # Prepare directory
    save_path = "ml_tflm/training/results/train_metrics.json"
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    # Find best result (highest macro F1)
    best_result = max(train_metrics, key=lambda d: d.get("f1", -1))

    # Append all history
    best_result["history"] = {k: utils.clean_metrics(v) for k, v in train_history.items()}
    print(train_history)

    # Dump result in a JSON-safe way
    with open(save_path, "w") as f:
        json.dump(to_serializable(best_result), f, indent=2)


if __name__ == "__main__":
    main()