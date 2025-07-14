import tensorflow as tf
from tensorflow.keras import layers, Model
from ml_tflm.models_tf.eegnet import EEGNet  # Your TensorFlow EEGNet
from ml_tflm.models_tf.attention_pooling import AttentionPooling  # TF version with flexible activation
from ml_tflm.models_tf.eegnet_Mod import EEGNetMod1

class EEGNetHierarchicalClassifier(Model):
    def __init__(self, eegnet_args, pooling_args=None, hidden_dim=128, num_classes=(2, 2, 2)):
        super().__init__()
        self.eegnet = EEGNet(**eegnet_args)
        pooling_args = pooling_args or {}
        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]
        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        self.classifier1 = layers.Dense(num_classes[0])
        self.classifier2 = layers.Dense(num_classes[1])
        self.classifier3 = layers.Dense(num_classes[2])

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, use_attention=False, training=False):
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * E, C, T, 1])
        epoch_features = self.eegnet(x, training=training)
        D = tf.shape(epoch_features)[-1]
        epoch_features = tf.reshape(epoch_features, [B, E, D])

        if use_attention:
            pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)
        else:
            if attention_mask is not None:
                mask_exp = tf.expand_dims(tf.cast(attention_mask, epoch_features.dtype), -1)
                pooled = tf.reduce_sum(epoch_features * mask_exp, axis=1) / tf.reduce_sum(mask_exp, axis=1)
            else:
                pooled = tf.reduce_mean(epoch_features, axis=1)
            attn_weights = tf.ones([B, E], dtype=epoch_features.dtype) / tf.cast(E, epoch_features.dtype)

        out = {
            "level1_logits": self.classifier1(pooled),
            "level2_logits": self.classifier2(pooled),
            "level3_logits": self.classifier3(pooled)
        }
        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled
        return out

class EEGNetFlatClassifier(Model):
    def __init__(self, eegnet_args, pooling_args=None, num_classes=4):
        super().__init__()
        self.eegnet = EEGNet(**eegnet_args)
        pooling_args = pooling_args or {}
        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]
        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        self.classifier = layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, use_attention=False, training=False):
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * E, C, T, 1])
        epoch_features = self.eegnet(x, training=training)
        D = tf.shape(epoch_features)[-1]
        epoch_features = tf.reshape(epoch_features, [B, E, D])

        if use_attention:
            pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)
        else:
            if attention_mask is not None:
                mask_exp = tf.expand_dims(tf.cast(attention_mask, epoch_features.dtype), -1)
                pooled = tf.reduce_sum(epoch_features * mask_exp, axis=1) / tf.reduce_sum(mask_exp, axis=1)
            else:
                pooled = tf.reduce_mean(epoch_features, axis=1)
            attn_weights = tf.ones([B, E], dtype=epoch_features.dtype) / tf.cast(E, epoch_features.dtype)

        logits = self.classifier(pooled)
        out = {"logits": logits}
        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled
        return out
    
    def build(self, input_shape):
        super().build(input_shape)


class EEGNetFlatClassifierMod1(Model):
    def __init__(self, eegnet_args, pooling_args=None, num_classes=4):
        super().__init__()
        self.eegnet = EEGNetMod1(**eegnet_args)
        pooling_args = pooling_args or {}
        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]
        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        self.classifier = layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, use_attention=False, training=False):
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * E, C, T, 1])
        epoch_features = self.eegnet(x, training=training)
        D = tf.shape(epoch_features)[-1]
        epoch_features = tf.reshape(epoch_features, [B, E, D])

        if use_attention:
            pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)
        else:
            if attention_mask is not None:
                mask_exp = tf.expand_dims(tf.cast(attention_mask, epoch_features.dtype), -1)
                pooled = tf.reduce_sum(epoch_features * mask_exp, axis=1) / tf.reduce_sum(mask_exp, axis=1)
            else:
                pooled = tf.reduce_mean(epoch_features, axis=1)
            attn_weights = tf.ones([B, E], dtype=epoch_features.dtype) / tf.cast(E, epoch_features.dtype)

        logits = self.classifier(pooled)
        out = {"logits": logits}
        if return_attn_weights:
            out["attention_weights"] = attn_weights
        if return_features:
            out["features"] = pooled
        return out
    
    def build(self, input_shape):
        super().build(input_shape)
