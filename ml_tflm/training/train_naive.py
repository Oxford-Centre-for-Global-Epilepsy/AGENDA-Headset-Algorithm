from ml_tflm.models_tf.classifiers import EEGNetFlatClassifier
from ml_tflm.dataset.eeg_dataset import EEGRecordingTFGenerator
import ml_tflm.training.train_utils as utils
from ml_tflm.training.loss import StructureAwareLoss

import numpy as np
import tensorflow as tf
import os

if __name__ == "__main__":
    

    # Load label configuration
    label_config = utils.load_label_config("ml_tflm/training/label+map.JSON")

    # Load EEG datasets
    train_dataset, val_dataset, test_dataset = utils.load_eeg_datasets_split(
        h5_file_path="ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5",
        dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20",
        label_map=label_config["label_map"],
    )
    
    # Create the loss evalutaion function
    loss_fn = StructureAwareLoss(
        label_config=label_config,
        temperature=5.0,
    )

    