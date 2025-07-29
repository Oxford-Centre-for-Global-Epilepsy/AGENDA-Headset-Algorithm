import tensorflow as tf

from ml_tflm.models_tf.feature_extractor import EEGNet
from ml_tflm.models_tf.attention_pooling import GetPoolingLayer
from ml_tflm.models_tf.classification_head import GetClassifierHead

class EEGNetStack(tf.keras.Model):
    """
    Modular EEG classification stack that combines:
    - A feature extractor (e.g., EEGNet or custom encoder)
    - An optional pool layer (e.g., MultiHeadAttentionpool)
    - A classifier head (flat or hierarchical)

    This class assumes all input pre-processing (e.g., reshaping from [B, E, C, T]) is handled by the components.
    """

    def __init__(self, eegnet_args, classifier_head_args, pool_args=None):
        """
        Args:
            eegnet: A module that outputs either [B, D] or [B, E, D] depending on pool
            classifier_head: A flat or hierarchical classifier head, applied after pool
            pool: Optional pool layer (e.g., MultiHeadAttentionpool or masked mean pool)
        """
        super().__init__()

        self.eegnet = EEGNet(**eegnet_args)

        if pool_args:
            dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])
            feature_dim = self.eegnet(dummy_input, training=False).shape[-1]
            self.pool = GetPoolingLayer(input_dim=feature_dim, **pool_args)
        else:
            self.pool = None

        self.classifier = GetClassifierHead(**classifier_head_args)

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, use_attention=False, training=False):
        # Input shape: [B, E, C, T]
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        # Flatten across epochs → [B * E, C, T, 1]
        x = tf.reshape(x, [B * E, C, T, 1])
        epoch_features = self.eegnet(x, training=training)  # → [B * E, D]
        D = tf.shape(epoch_features)[-1]

        # Restore epoch dimension → [B, E, D]
        epoch_features = tf.reshape(epoch_features, [B, E, D])

        if use_attention:
            # Apply attention pool
            pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)
        else:
            # Fallback to mean or masked mean
            if attention_mask is not None:
                mask_exp = tf.expand_dims(tf.cast(attention_mask, dtype=epoch_features.dtype), axis=-1)
                pooled = tf.reduce_sum(epoch_features * mask_exp, axis=1) / tf.reduce_sum(mask_exp, axis=1)
            else:
                pooled = tf.reduce_mean(epoch_features, axis=1)

            # Uniform dummy weights
            attn_weights = tf.ones([B, E], dtype=epoch_features.dtype) / tf.cast(E, epoch_features.dtype)

        # Final classification
        out = self.classifier(pooled)

        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled

        return out
