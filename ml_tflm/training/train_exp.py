import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
from omegaconf import DictConfig
import tensorflow as tf
from ml_tflm.training.trainer import Trainer
import ml_tflm.training.train_utils as utils
from hydra.utils import instantiate


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # --- Load label config ---
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    # --- Load dataset ---
    train_val_sets, test_dataset = utils.load_eeg_datasets_split(
        h5_file_path=cfg.dataset.h5_path,
        dataset_name=cfg.dataset.name,
        label_config=label_config,
        val_frac=cfg.training.val_frac,
        test_frac=cfg.training.test_frac,
        k_fold=cfg.training.k_fold
    )

    for train_val_set in train_val_sets:
        train_dataset = train_val_set[0]
        val_dataset = train_val_set[1]

        # --- Prepare class histogram and loss ---
        class_hist = utils.compute_label_histogram(train_dataset, label_config)
        loss_fn = instantiate(cfg.architecture.loss, label_config=label_config, class_histogram=class_hist)

        # --- Get EEGNet shape info ---
        data_spec = train_dataset.element_spec["data"]
        _, E, C, T = data_spec.shape

        # --- Resolve model args ---
        eegnet_args = dict(cfg.architecture.model.eegnet_args)
        eegnet_args["num_channels"] = C
        eegnet_args["num_samples"] = T
        eegnet_args["activation"] = getattr(tf.nn, eegnet_args["activation"])

        pooling_args = dict(cfg.architecture.model.pooling_args)
        pooling_args["activation"] = getattr(tf.nn, pooling_args["activation"])

        model = instantiate(cfg.architecture.model, eegnet_args=eegnet_args, pooling_args=pooling_args)

        # --- Optimizer ---
        optimizer = instantiate(cfg.optimizer)
        
        # --- Evaluator ---
        evaluator = instantiate(
            cfg.architecture.evaluator,
            label_config=label_config,
        )

        # --- Trainer ---
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            evaluator=evaluator,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model_input_lookup=cfg.architecture.model_input_lookup,
            model_target_lookup=cfg.architecture.model_target_lookup,
        )

        # --- Train ---
        trainer.train_loop(
            epochs=cfg.training.epochs,
            steps_per_epoch=cfg.training.steps_per_epoch
        )


if __name__ == "__main__":
    main()
