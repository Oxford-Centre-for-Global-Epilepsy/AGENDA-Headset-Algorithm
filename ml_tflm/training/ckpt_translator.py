"""
This script is intended only for converting .ckpt files into full models. 
For the remaining conversion steps, please refer to the ~/model_conversion_factory directory.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
from omegaconf import DictConfig
import tensorflow as tf
from hydra.utils import instantiate
import ml_tflm.training.train_utils as utils

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    This script is intended only for converting .ckpt files into full models.
    For the rest of the conversion process, please refer to the ~/conversion directory.
    """

    # --- Load label config (used for consistent model construction) ---
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    # --- Load dataset to infer shape ---
    train_val_sets, _ = utils.load_eeg_datasets_split(
        h5_file_path=cfg.dataset.h5_path,
        dataset_name=cfg.dataset.name,
        label_config=label_config,
        val_frac=cfg.training.val_frac,
        test_frac=cfg.training.test_frac,
        k_fold=cfg.training.k_fold
    )
    train_dataset = train_val_sets[0][0]  # Only need one for shape

    # --- Extract shape ---
    data_spec = train_dataset.element_spec["data"]
    _, E, C, T = data_spec.shape

    # --- Instantiate model ---
    eegnet_args = dict(cfg.architecture.model.eegnet_args)
    eegnet_args["num_channels"] = C
    eegnet_args["num_samples"] = T
    eegnet_args["activation"] = getattr(tf.nn, eegnet_args["activation"])

    pooling_args = dict(cfg.architecture.model.pooling_args)
    pooling_args["activation"] = getattr(tf.nn, pooling_args["activation"])

    model = instantiate(cfg.architecture.model, eegnet_args=eegnet_args, pooling_args=pooling_args)

    dummy_input = tf.zeros([1, 5, C, T])
    _ = model(dummy_input)

    # --- Restore from checkpoint ---
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(cfg.training.ckpt_load_dir)
    if latest_ckpt:
        ckpt.restore(latest_ckpt).expect_partial()
        print(f"Restored checkpoint from {latest_ckpt}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {cfg.training.ckpt_load_dir}")

    # --- Save model in .h5 format (for further conversion/export) ---
    output_path = "ml_tflm/model_conversion_factory/model_FULL"
    model.save(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    main()
