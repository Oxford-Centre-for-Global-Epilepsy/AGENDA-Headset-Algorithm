import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_model_optimization as tfmot

import logging
tf.get_logger().setLevel(logging.ERROR)

"""
MESSAGE: Due to behavior of Tensorflow NOT UNDERSTOOD YET, all model components are contained in this folder
"""

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

class AttentionPooling(tf.keras.layers.Layer):
    """
    Learnable attention pooling over epochs.

    Parameters
    ----------
    input_dim : int
        Feature dimension of each epoch after EEGNet.
    hidden_dim : int
        Hidden layer size for computing attention scores.
    activation : tf.keras activation function
        Non-linearity used between dense layers (default = tf.nn.tanh).
    """
    def __init__(self, input_dim, hidden_dim=128, activation=tf.nn.tanh):
        super().__init__()
        self.activation = activation
        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, mask=None, return_weights=False):
        """
        Parameters
        ----------
        x : Tensor of shape [B, E, D]
            Batch of EEGNet features over epochs.
        mask : Boolean Tensor of shape [B, E], optional
            Attention mask indicating valid epochs.
        return_weights : bool
            If True, also return attention weights.

        Returns
        -------
        Tensor of shape [B, D] or tuple (pooled, weights)
        """
        scores = self.dense2(self.activation(self.dense1(x)))  # [B, E, 1]
        scores = tf.squeeze(scores, axis=-1)  # [B, E]

        if mask is not None:
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

        weights = tf.nn.softmax(scores, axis=1)  # [B, E]
        weights_expanded = tf.expand_dims(weights, axis=-1)  # [B, E, 1]
        pooled = tf.reduce_sum(x * weights_expanded, axis=1)  # [B, D]

        if return_weights:
            return pooled, weights

        return pooled

class EEGNetFlatClassifierQAT(tf.keras.Model):
    def __init__(self,
                 eegnet_qat,
                 pooling_args=None,
                 num_classes=4):  # Only valid class combinations
        """
        EEGNet-based classifier using a single softmax over 4 valid classes.

        Args:
            eegnet_args (dict): kwargs to initialize EEGNet
            pooling_args (dict): kwargs for AttentionPooling
            num_classes (int): number of valid hierarchical class combinations
        """
        super().__init__()
        self.eegnet = eegnet_qat
        pooling_args = pooling_args or {}

        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])  # [B, C, T, 1] for EEGNet
        feature_dim = self.eegnet(dummy_input).shape[-1]

        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        self.classifier = tf.keras.layers.Dense(num_classes)  # No activation

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, training=False):
        """
        Args:
            x: [B, E, C, T]
            attention_mask: [B, E]
            return_attn_weights: if True, returns attention weights
            return_features: if True, returns pooled features

        Returns:
            dict with logits and optional weights/features
        """
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * E, C, T, 1]) # matches the channel-last format expected by EEGNet
        epoch_features = self.eegnet(x, training=training)
        D = tf.shape(epoch_features)[-1]
        epoch_features = tf.reshape(epoch_features, [B, E, D])

        pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)
        logits = self.classifier(pooled)

        out = {"logits": logits}
        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled

        return out

def get_classifier_QAT(eegnet_args, pooling_args):
    qat_model = get_eegnet_qat(eegnet_args)
    return EEGNetFlatClassifierQAT(eegnet_qat=qat_model,
                                   pooling_args=pooling_args,
                                   num_classes=4)

if __name__ == "__main__":
    eegnet_args = {
        "num_channels": 21,
        "num_samples": 256,
        "F1": 8,
        "D": 2,
        "F2": 16,
        "dropout_rate": 0.25,
        "kernel_length": 64,
        "activation": tf.nn.relu
    }

    pooling_args = {
        "hidden_dim": 64,
        "activation": tf.nn.tanh
    }

    qat_model = get_classifier_QAT(eegnet_args, pooling_args)
    
    print("Quantisation Successful!")