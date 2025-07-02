from ml_tflm.model_conversion_factory.conversion_utils import convert_to_tflite, convert_tflite_to_micro
from ml_tflm.model_conversion_factory.representative_dataset import get_rep_dataset
from ml_tflm.models_tf.classifier_QAT import AttentionPooling  # adjust import path
import tensorflow as tf
import tensorflow_model_optimization as tfmot

if __name__ == "__main__":
    # --- Paths ---
    eegnet_path = "ml_tflm/model_conversion_factory/model_SPLIT/model_FEATURE_EXTRACTOR.keras"
    classifier_path = "ml_tflm/model_conversion_factory/model_SPLIT/model_CLASSIFIER_HEAD.keras"
    tflite_eegnet_path = "ml_tflm/model_conversion_factory/model_park/model_eegnet_quant_test01.tflite"
    micro_eegnet_cc_path = "ml_tflm/model_conversion_factory/model_park/model_eegnet_quant_test01.cc"
    tflite_classifier_path = "ml_tflm/model_conversion_factory/model_park/model_classifier_test01.tflite"

    # --- Load models ---
    print("Loading models...")
    with tfmot.quantization.keras.quantize_scope():
        eegnet_model = tf.keras.models.load_model(eegnet_path)

    classifier_model = tf.keras.models.load_model(classifier_path)

    # --- Create representative dataset for EEGNet quantization ---
    print("Creating representative dataset...")
    rep_dataset = get_rep_dataset(
        h5_file_path="ml_tflm/dataset/sample_data/anyu_dataset_south_africa_monopolar_standard_10_20.h5",
        dataset_name="anyu_dataset_south_africa_monopolar_standard_10_20",
        total_epochs=100,
        epochs_per_subject=3
    )

    # --- Convert EEGNet model with quantization ---
    print("Converting EEGNet model to quantized TFLite...")
    convert_to_tflite(
        model=eegnet_model,
        save_path=tflite_eegnet_path,
        rep_dataset=rep_dataset
    )

    # --- Convert to TFLite Micro .cc file ---
    print("Converting to TFLite Micro .cc format...")
    convert_tflite_to_micro(
        tflite_model_path=tflite_eegnet_path,
        output_cc_path=micro_eegnet_cc_path,
        var_name="g_eegnet_model"
    )

    # --- Convert classifier model to plain TFLite ---
    print("Converting classifier model to TFLite (no quantization)...")
    convert_to_tflite(
        model=classifier_model,
        save_path=tflite_classifier_path,
        rep_dataset=None  # no quantization
    )

    print("Conversion complete.")
