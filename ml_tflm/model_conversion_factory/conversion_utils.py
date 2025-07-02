import tensorflow as tf

def convert_to_tflite(model, save_path, rep_dataset=None, optimizations=None):
    """
    Converts a Keras model to TFLite format.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model to convert.
    save_path : str
        Path to save the converted .tflite model.
    rep_dataset : EEGRepresentativeDataset, optional
        Representative dataset for quantization. If None, converts without quantization.
    optimizations : list of tf.lite.Optimize, optional
        List of optimizations to apply. Defaults to [tf.lite.Optimize.DEFAULT] if rep_dataset is given.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if rep_dataset is not None:
        converter.optimizations = optimizations or [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_dataset.generator
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    with open(save_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {save_path}")

def convert_tflite_to_micro(tflite_model_path, output_cc_path, var_name="g_model"):
    """
    Converts a .tflite model to a C source file for TFLite Micro.

    Parameters
    ----------
    tflite_model_path : str
        Path to the .tflite model file.
    output_cc_path : str
        Path to the output .cc file.
    var_name : str
        C variable name to use for the model.
    """
    with open(tflite_model_path, "rb") as f:
        model_bytes = f.read()

    with open(output_cc_path, "w") as f:
        f.write(f'#include "tensorflow/lite/micro/micro_model_settings.h"\n\n')
        f.write(f'const unsigned char {var_name}[] = {{\n')

        for i, byte in enumerate(model_bytes):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f'const int {var_name}_len = {len(model_bytes)};\n')

    print(f"TFLite Micro model saved to {output_cc_path}")

