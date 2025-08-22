from ml_tflm.model_conversion_factory.conversion_utils import convert_to_tflite, convert_tflite_to_micro
from ml_tflm.model_conversion_factory.representative_dataset import get_rep_dataset
from ml_tflm.models_tf.classifier_QAT import AttentionPooling, AveragePooling, HardSwish  # adjust import path
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from ml_tflm.training.ckpt_translator import loadTrainedModel

if __name__ == "__main__":
    # --- Paths ---
    eegnet_path = "ml_tflm/model_conversion_factory/model_SPLIT/model_FEATURE_EXTRACTOR.keras"
    classifier_path = "ml_tflm/model_conversion_factory/model_SPLIT/model_CLASSIFIER_HEAD.keras"
    tflite_eegnet_path = "ml_tflm/model_conversion_factory/model_park/model_eegnet_quant_test01.tflite"
    micro_eegnet_cc_path = "ml_tflm/model_conversion_factory/model_park/model_eegnet_quant_test01.cc"
    tflite_classifier_path = "ml_tflm/model_conversion_factory/model_park/model_classifier_test01.tflite"
    micro_classifier_cc_path = "ml_tflm/model_conversion_factory/model_park/model_classifier_test01.cc"

    # --- Load models ---
    print("Loading models...")
    # with tfmot.quantization.keras.quantize_scope():
    #     eegnet_model = tf.keras.models.load_model(eegnet_path)

    # with tfmot.quantization.keras.quantize_scope():
    #     classifier_model = tf.keras.models.load_model(classifier_path)

    eegnet_model, classifier_model = loadTrainedModel(save=False)

    # --- Create representative dataset for EEGNet quantization ---
    print("Creating representative dataset...")
    rep_dataset = get_rep_dataset(h5_file_path="ml_tflm/dataset/agenda_data_23_bp45_tr05/merged_south_africa_monopolar_standard_10_20.h5",
                                  dataset_name="combined_south_africa_monopolar_standard_10_20",
                                  total_epochs=100, epochs_per_subject=2)

    # --- Convert EEGNet model with quantization ---
    print("Converting EEGNet model to quantized TFLite...")
    convert_to_tflite(
        model=eegnet_model,
        save_path=tflite_eegnet_path,
        rep_dataset=rep_dataset
    )

    # --- Convert classifier model to plain TFLite ---
    print("Converting classifier model to TFLite (no quantization)...")
    convert_to_tflite(
        model=classifier_model,
        save_path=tflite_classifier_path,
        rep_dataset=None  # no quantization
    )

    # --- Convert the TFLite models to flatbuffer ---
    print("Converting TFLite models to FlatBuffer")
    convert_tflite_to_micro(tflite_eegnet_path, micro_eegnet_cc_path)
    convert_tflite_to_micro(tflite_classifier_path, micro_classifier_cc_path)


    print("Conversion complete.")
