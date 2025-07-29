import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_LAYOUT_OPTIMIZER"] = "0"
os.environ["TF_AUTOGRAPH_CACHE_DIR"] = "D:/tf_autograph"

import hydra
from omegaconf import DictConfig
import tensorflow as tf

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

from ml_tflm.training.trainer import Trainer
import ml_tflm.training.train_utils as utils

from ml_tflm.dataset.eeg_dataset_tfrecord import load_dataset_meta_or_infer, load_eeg_tfrecord_dataset
from hydra.utils import instantiate

from ml_tflm.models_tf.eegnet_stack import EEGNetStack

import json
from pathlib import Path
import numpy as np


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

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    print("--- Training Starting ---")

    # --- Load label config ---
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    print(" -> Label Map Loaded")

    if cfg.training.internal_label_cap_keys and cfg.training.internal_label_cap_values:
        internal_label_cap = dict(zip(cfg.training.internal_label_cap_keys, cfg.training.internal_label_cap_values))
    else:
        internal_label_cap = None

    # --- Load dataset ---
    train_val_sets, test_dataset, label_histograms = utils.prepare_eeg_datasets(
        h5_file_path=cfg.dataset.h5_path,
        dataset_name=cfg.dataset.name,
        label_config=label_config,
        omit_channels=cfg.dataset.ablation,
        val_frac=cfg.training.val_frac,
        test_frac=cfg.training.test_frac,
        k_fold=cfg.training.k_fold,
        stratify=cfg.training.stratify,
        internal_label_cap=internal_label_cap,
        batch_size=cfg.training.batch_sz,
        mirror_flag=cfg.training.mirror_flag,
    )
    
    print(" -> Dataset Loaded")
    print("     -> Loaded Dataset Contains:")
    print(label_histograms)

    # --- Define metric holder ---
    train_metrics = []

    for fold_idx in range(len(train_val_sets)):
        train_val_set = train_val_sets[fold_idx]
        train_dataset = train_val_set[0]
        val_dataset = train_val_set[1]

        # --- Prepare class histogram and loss ---
        if cfg.training.class_balance:
            class_hist = label_histograms[fold_idx]
        else:
            class_hist = None
        
        loss_fn = instantiate(cfg.component.classifier_head.loss, label_config=label_config, class_histogram=class_hist)
        entropy_loss = instantiate(cfg.component.pooling_layer.loss) if "loss" in cfg.component.pooling_layer else None

        # --- Get EEGNet shape info ---
        data_spec = train_dataset.element_spec["data"]
        _, E, C, T = data_spec.shape

        # --- Resolve model args ---
        eegnet_args = dict(cfg.component.feature_extractor.eegnet)
        eegnet_args["num_channels"] = C
        eegnet_args["num_samples"] = T
        eegnet_args["activation"] = utils.get_activation(eegnet_args["activation"])

        pooling_args = dict(cfg.component.pooling_layer.pool)
        if "activation" in pooling_args:
            pooling_args["activation"] = getattr(tf.nn, pooling_args["activation"])

        head_args = dict(cfg.component.classifier_head.head)

        model = EEGNetStack(
            eegnet_args=eegnet_args,
            classifier_head_args=head_args,
            pool_args=pooling_args
        )
        
        # Now create optimizer after variables exist
        optimizer = instantiate(cfg.optimizer)

        # --- Evaluator ---
        evaluator = instantiate(
            cfg.component.classifier_head.evaluator,
            label_config=label_config,
        )

        # --- Trainer ---
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            attention_loss=entropy_loss,
            evaluator=evaluator,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_input_lookup=cfg.component.feature_extractor.model_input_lookup,
            model_target_lookup=cfg.component.classifier_head.model_target_lookup,
            save_ckpt=cfg.training.save_ckpt,
            ckpt_interval=cfg.training.ckpt_interval,
            ckpt_save_dir=cfg.training.ckpt_save_dir,
            load_ckpt=cfg.training.load_ckpt,
            ckpt_load_dir=cfg.training.ckpt_load_dir,
            attention_warmup_epoch=cfg.training.attention_warmup_epoch,
            anneal_interval=cfg.training.anneal_interval,
            anneal_coeff=cfg.training.anneal_coeff
        )

        print(" -> Training Time!")


        # --- Train ---
        trainer.train_loop(
            epochs=cfg.training.epochs,
            steps_per_epoch=cfg.training.steps_per_epoch
        )

        train_metrics.append(trainer.get_metrics())

    # Prepare directory
    save_path = cfg.training.metric_save_dir
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

    # Find best result (highest macro F1)
    best_result = max(train_metrics, key=lambda d: d.get("f1", -1))

    # Dump result in a JSON-safe way
    with open(save_path, "w") as f:
        json.dump(to_serializable(best_result), f, indent=2)


if __name__ == "__main__":
    main()