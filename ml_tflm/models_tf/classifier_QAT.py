import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.quantization.keras import quantize_annotate_layer as QAnnotate

import logging
tf.get_logger().setLevel(logging.ERROR)

QK = tfmot.quantization.keras
Quantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

"""
MESSAGE: Due to behavior of Tensorflow NOT UNDERSTOOD YET, all model components are contained in this folder
"""

# ##################
# ##### CUSTOM #####
# ##################

class HardSwishQuantizeConfig(QK.QuantizeConfig):
    """Quantize the OUTPUT of HardSwish only (8-bit activations)."""
    def get_weights_and_quantizers(self, layer): return []   # no weights in HardSwish
    def get_activations_and_quantizers(self, layer): return []
    def set_quantize_weights(self, layer, quantize_weights): pass
    def set_quantize_activations(self, layer, quantize_activations): pass
    def get_output_quantizers(self, layer):
        # IMPORTANT: keep it simple; no 'name' arg or extras
        return [Quantizer(
            num_bits=8, per_axis=False, symmetric=False, narrow_range=False
        )]
    def get_config(self): return {}

@tf.keras.utils.register_keras_serializable(package="Compat")
class HardSwish(tf.keras.layers.Layer):
    def call(self, x):
        # y = x * relu6(x + 3) * (1/6), no division op
        c3   = tf.cast(3.0,     x.dtype)
        c1_6 = tf.cast(1.0/6.0, x.dtype)
        return x * tf.nn.relu6(x + c3) * c1_6

    def get_config(self):
        return {**super().get_config()}

# ##################
# ##### EEGNET #####
# ##################

def get_eegnet_model(
    num_channels=16, num_samples=256,
    dropout_rate=0.5, F1=16, D=2, F2=4,
    kernel_length=64
):
    inputs = tf.keras.Input(shape=(num_channels, num_samples, 1))  # [B, C, T, 1]

    # Block 1: Temporal Conv
    x = tf.keras.layers.Conv2D(F1, (1, kernel_length), padding='same', use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    temporal_features = x  # (kept if you use elsewhere)

    # Block 2: Depthwise Spatial Conv
    x = tf.keras.layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = QAnnotate(HardSwish(name="spatial_hswish"), quantize_config=HardSwishQuantizeConfig())(x)
    x = tf.keras.layers.AveragePooling2D((1, 4))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    spatial_features = x

    # Block 3: Separable Conv
    x = tf.keras.layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = QAnnotate(HardSwish(name="sep_hswish"), quantize_config=HardSwishQuantizeConfig())(x)
    x = tf.keras.layers.AveragePooling2D((1, 8))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    final_features = x

    # Flatten
    features = tf.keras.layers.Flatten()(final_features)

    return tf.keras.Model(inputs=inputs, outputs=features, name="EEGNet_Functional")

def get_eegnet_qat(eegargs):
    base = get_eegnet_model(**eegargs)

    registry = QK.default_8bit.Default8BitQuantizeRegistry()

    # Clone-time annotator: annotate supported layers; keep existing QuantizeAnnotate as-is
    def _annotate_or_keep(layer):
        # Keep your already-annotated HardSwish markers untouched
        if layer.__class__.__name__ == 'QuantizeAnnotate':
            return layer
        # Annotate default-quantizable layers (Conv/Depthwise/Separable/Dense/ReLU, etc.)
        try:
            if registry.supports(layer):
                # ensure layer has a name (Keras 3 sometimes leaves it blank)
                if not getattr(layer, "name", None):
                    layer._name = layer.__class__.__name__.lower()
                return QK.quantize_annotate_layer(layer)
        except Exception:
            pass  # leave unsupported/no-op layers as-is (Pooling/Dropout/Flatten/etc.)
        return layer

    with QK.quantize_scope({
        'HardSwish': HardSwish,
        'HardSwishQuantizeConfig': HardSwishQuantizeConfig
    }):
        annotated = tf.keras.models.clone_model(base, clone_function=_annotate_or_keep)
        qat_model = QK.quantize_apply(annotated)

    return qat_model

# ################
# ##### POOL #####
# ################

@tf.keras.utils.register_keras_serializable()
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
    def __init__(self, input_dim, hidden_dim=128, activation=tf.nn.tanh, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.dense1 = tf.keras.layers.Dense(hidden_dim)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, mask=None):
        scores = self.dense2(self.activation(self.dense1(x)))  # [B, E, 1]
        scores = tf.squeeze(scores, axis=-1)  # [B, E]

        if mask is not None:
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

        weights = tf.nn.softmax(scores, axis=1)  # [B, E]
        weights_expanded = tf.expand_dims(weights, axis=-1)  # [B, E, 1]
        pooled = tf.reduce_sum(x * weights_expanded, axis=1)  # [B, D]
        return pooled

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "activation": tf.keras.activations.serialize(self.activation)
        }

    @classmethod
    def from_config(cls, config):
        config["activation"] = tf.keras.activations.deserialize(config["activation"])
        return cls(**config)

def get_attention_classifier(feature_dim, pooling_args, num_classes=4, name="AttentionClassifier"):
    """
    Create a Keras Functional model that accepts feature sequence and optional attention mask.

    Inputs:
    - feature_sequence: [B, E, D]
    - attention_mask:   [B, E] (bool mask)

    Output:
    - logits:           [B, num_classes]
    """
    # Inputs
    feature_input = tf.keras.Input(shape=(None, feature_dim), name="feature_sequence")   # [B, E, D]
    mask_input = tf.keras.Input(shape=(None,), dtype=tf.bool, name="attention_mask")     # [B, E]

    # Apply attention pooling with mask
    pooling_layer = AttentionPooling(input_dim=feature_dim, **pooling_args)
    x = pooling_layer(feature_input, mask=mask_input)

    # Final classifier layer
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)

    # Create model with both inputs
    return tf.keras.Model(inputs=[feature_input, mask_input], outputs=logits, name=name)

@tf.keras.utils.register_keras_serializable()
class AveragePooling(tf.keras.layers.Layer):
    """
    Simple average pooling over epochs, mask-aware.

    Parameters
    ----------
    None
        (No trainable parameters; pure reduction)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, mask=None):
        """
        x:    [B, E, D]  feature sequence
        mask: [B, E]     boolean mask (True = keep, False = ignore)
        """
        if mask is not None:
            # Cast mask to float32 for multiplication
            mask_f = tf.cast(mask, tf.float32)  # [B, E]
            mask_f_exp = tf.expand_dims(mask_f, axis=-1)  # [B, E, 1]
            x_masked = x * mask_f_exp

            sum_vec = tf.reduce_sum(x_masked, axis=1)  # [B, D]
            count_vec = tf.reduce_sum(mask_f, axis=1, keepdims=True)  # [B, 1]
            pooled = sum_vec / tf.maximum(count_vec, 1.0)  # avoid div-by-zero
        else:
            pooled = tf.reduce_mean(x, axis=1)  # [B, D]
        return pooled

    def get_config(self):
        return super().get_config()

def get_average_pooling_model(feature_dim, name="AveragePoolingModel"):
    """
    Create a Keras Functional model that accepts feature sequence and optional mask,
    returning only the pooled vector (no classifier).
    """
    feature_input = tf.keras.Input(shape=(None, feature_dim), name="feature_sequence")  # [B, E, D]
    mask_input = tf.keras.Input(shape=(None,), dtype=tf.bool, name="attention_mask")    # [B, E]

    pooling_layer = AveragePooling()
    pooled_output = pooling_layer(feature_input, mask=mask_input)

    return tf.keras.Model(inputs=[feature_input, mask_input], outputs=pooled_output, name=name)

# ################
# ##### HEAD #####
# ################

def get_flat_classifier_head(num_classes=4, l2_weight=1e-5):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            num_classes,
            activation=None,  # logits
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
            name="flat_classifier_dense"
        )
    ], name="flat_classifier_head")

def get_head_qat(headargs, feature_dim):
    # Option A: build Sequential then quantize
    model = get_flat_classifier_head(**headargs)   # your Sequential(Dense)
    model.build((None, feature_dim))               # make it "built"

    quantise_model = tfmot.quantization.keras.quantize_model
    return quantise_model(model)

# #################
# ##### MODEL #####
# #################

class EEGNetFlatClassifierQAT(tf.keras.Model):
    def __init__(self,
                 eegnet_args,
                 head_args):
        """
        EEGNet-based classifier using a single softmax over 4 valid classes.

        Args:
            eegnet_args (dict): kwargs to initialize EEGNet
            pooling_args (dict): kwargs for AttentionPooling
            num_classes (int): number of valid hierarchical class combinations
        """
        super().__init__()
        # Instantiate and store quantized feature extractor
        self.eegnet = get_eegnet_qat(eegargs=eegnet_args)

        # Probe feature dim by running dummy input
        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])
        feature_dim = self.eegnet(dummy_input).shape[-1]

        # Define pooling + classifier head
        self.pool = get_average_pooling_model(feature_dim)

        # self.classifier = get_head_qat(head_args, feature_dim)
        self.classifier = get_flat_classifier_head(**head_args)
        self.classifier.build((None, feature_dim))

    def call(self, x, attention_mask=None, use_attention=None, return_attn_weights=None, return_features=None, training=False):
        """
        Args:
            x: Tensor [B, E, C, T]
            attention_mask: optional bool mask [B, E]

        Returns:
            {"logits": [B, num_classes]}
        """
        B = tf.shape(x)[0]
        E = tf.shape(x)[1]
        C = tf.shape(x)[2]
        T = tf.shape(x)[3]

        # 1) Flatten epochs -> per-epoch features
        x_flat = tf.reshape(x, [B * E, C, T, 1])                     # [B*E, C, T, 1]
        feats  = self.eegnet(x_flat, training=training)    # [B*E, D]

        # 2) Restore [B, E, D]
        D = tf.shape(feats)[-1]
        feats = tf.reshape(feats, [B, E, D])                          # [B, E, D]

        # 3) Pool over epochs (default: all valid)
        if attention_mask is None:
            attention_mask = tf.ones([B, E], dtype=tf.bool)

        pooled = self.pool([feats, attention_mask], training=False)   # [B, D]

        # 4) Head -> logits
        logits = self.classifier(pooled, training=training)                 # [B, num_classes]

        return {"logits": logits}


if __name__ == "__main__":
    eegnet_args = {
        "num_channels": 16,
        "num_samples": 128,
        "F1": 16,
        "D": 2,
        "F2": 4,
        "dropout_rate": 0.25,
        "kernel_length": 64,
    }

    head_args = {
        "num_classes": 2,
        "l2_weight": 1e-5
    }

    model = EEGNetFlatClassifierQAT(eegnet_args=eegnet_args,
                            head_args=head_args)

    print("Quantisation Successful!")