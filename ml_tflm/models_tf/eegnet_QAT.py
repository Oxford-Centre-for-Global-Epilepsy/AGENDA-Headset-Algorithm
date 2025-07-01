import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_model_optimization as tfmot

import logging
tf.get_logger().setLevel(logging.ERROR)

def get_eegnet_model(
    num_channels = 21, num_samples = 256,
    dropout_rate = 0.5, F1 = 8, D = 2, F2 = 16, 
    kernel_length = 64, activation = tf.nn.elu
):
    """
    Functional EEGNet model for feature extraction.

    Parameters
    ----------
    input_shape : tuple
        Shape of input (channels, samples), not including batch or channel-last dim.
    dropout_rate : float
        Dropout rate.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier for spatial filters.
    F2 : int
        Number of pointwise filters.
    kernel_length : int
        Temporal kernel length.
    activation : callable
        Activation function.

    Returns
    -------
    tf.keras.Model
        Keras functional model.
    """
    inputs = tf.keras.Input(shape=(num_channels, num_samples, 1))  # shape: [B, C, T, 1]

    # Block 1: Temporal Conv
    x = tf.keras.layers.Conv2D(F1, (1, kernel_length), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    temporal_features = x

    # Block 2: Depthwise Spatial Conv
    x = tf.keras.layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.AveragePooling2D((1, 4))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    spatial_features = x

    # Block 3: Separable Conv
    x = tf.keras.layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.AveragePooling2D((1, 8))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    final_features = x

    # Flatten
    features = tf.keras.layers.Flatten()(final_features)

    return tf.keras.Model(inputs=inputs, outputs=features, name="EEGNet_Functional")

def get_eegnet_qat(eegargs):
    model = get_eegnet_model(**eegargs)

    quantise_model = tfmot.quantization.keras.quantize_model
    return quantise_model(model)

if __name__ == "__main__":
    eegargs = {
        "num_channels": 21, 
        "num_samples": 256,
        "dropout_rate": 0.5, 
        "F1": 8, 
        "D": 2, 
        "F2": 16, 
        "kernel_length": 64, 
        "activation": tf.nn.relu
    }

    qat_model = get_eegnet_qat(eegargs)

    print("Quantisation Successful!")