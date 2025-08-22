"""
This script is intended only for converting .ckpt files into full models. 
For the remaining conversion steps, please refer to the ~/model_conversion_factory directory.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import ml_tflm.training.train_utils as utils
from ml_tflm.models_tf.classifier_QAT import EEGNetFlatClassifierQAT


def loadTrainedModel(save=False):
    """
    This script is intended only for converting .ckpt files into full models.
    For the rest of the conversion process, please refer to the ~/conversion directory.
    """

    # --- Load label config (used for consistent model construction) ---
    label_config = utils.load_label_config("ml_tflm/training/label_map.JSON")

    # --- define internal label cap ---
    internal_label_cap = {
        0: 450,
        3: 260,
        4: 94,
        5: 96
    }

    # --- Load dataset to infer shape ---
    train_val_sets, _, label_histograms = utils.prepare_eeg_datasets(
        h5_file_path="ml_tflm/dataset/agenda_data_23_bp45_tr05/merged_south_africa_monopolar_standard_10_20.h5",
        dataset_name="combined_south_africa_monopolar_standard_10_20",
        label_config=label_config,
        omit_channels=["A1","A2", "Fz", "Pz", "Cz"],
        val_frac=0.2,
        test_frac=0.0,
        k_fold=False,
        stratify=True,
        internal_label_cap=internal_label_cap,
        batch_size=32,
        mirror_flag=False,
        chunk_size=256
    )

    print(" -> Dataset Loaded")
    print("     -> Loaded Dataset Contains:" + str(label_histograms))

    train_dataset = train_val_sets[0][0]

    # --- Extract shape ---
    data_spec = train_dataset.element_spec["data"]
    _, E, C, T = data_spec.shape

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

    dummy_input = tf.zeros([1, 5, C, T])
    _ = model(dummy_input)

    # --- Restore from checkpoint ---
    ckpt_dir = "ml_tflm/training/checkpoints_fold4"
    ckpt_path = os.path.join(ckpt_dir, "ckpt-16")

    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(ckpt_path).expect_partial()
    print(f"Restored checkpoint from {ckpt_path}")

    # --- Save separated model components as .keras files ---
    if save:
        output_dir = "ml_tflm/model_conversion_factory/model_SPLIT"

        # Feature extractor (for embedded/edge use)
        feature_path = os.path.join(output_dir, "model_FEATURE_EXTRACTOR.keras")
        #model.eegnet.save(feature_path, save_format="tf")
        model.eegnet.save(feature_path)

        print(f"Feature extractor saved to {feature_path}")

        # Classifier head (for mobile/host processing)
        classifier_path = os.path.join(output_dir, "model_CLASSIFIER_HEAD.keras")
        model.classifier.save(classifier_path, save_format="tf")
        print(f"Classifier head saved to {classifier_path}")

    return model.eegnet, model.classifier


if __name__ == "__main__":
    loadTrainedModel(save=True)
