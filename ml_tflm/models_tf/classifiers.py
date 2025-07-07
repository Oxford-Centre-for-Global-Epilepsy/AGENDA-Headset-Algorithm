import tensorflow as tf
from tensorflow.keras import layers, Model
from ml_tflm.models_tf.eegnet import EEGNet  # Your TensorFlow EEGNet
from ml_tflm.models_tf.attention_pooling import AttentionPooling  # TF version with flexible activation
from ml_tflm.models_tf.eegnet_Mod import EEGNetMod1

class EEGNetHierarchicalClassifier(Model):
    def __init__(self,
                 eegnet_args,
                 pooling_args=None,
                 hidden_dim=128,
                 num_classes=(2, 2, 2)):
        """
        EEGNet-based hierarchical classifier with AttentionPooling.

        Args:
            eegnet_args (dict): kwargs to initialize EEGNet (must include num_channels and num_samples)
            pooling_args (dict): kwargs for AttentionPooling layer
            hidden_dim (int): hidden dimension for internal attention MLP (in pooling)
            num_classes (tuple): number of classes at each classification level
        """
        super().__init__()
        self.eegnet = EEGNet(**eegnet_args)
        pooling_args = pooling_args or {}

        # Infer feature dimension using dummy input
        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])  # [B, C, T, 1] for EEGNet
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]

        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)

        self.classifier1 = layers.Dense(num_classes[0])
        self.classifier2 = layers.Dense(num_classes[1])
        self.classifier3 = layers.Dense(num_classes[2])

    def call(self, x, attention_mask=None, return_attn_weights=False, return_features=False, training=False):
        """
        Args:
            x: [B, E, C, T] input EEG batch
            attention_mask: [B, E] mask for valid epochs
            return_attn_weights: whether to return attention weights
            return_features: whether to return the pooled features
            training: True for dropout etc.

        Returns:
            dict with logits and optional attention/feature outputs
        """
        B, E, C, T = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, [B * E, C, T, 1])  # [B*E, 1, C, T]
        epoch_features = self.eegnet(x, training=training)  # [B*E, D]
        D = tf.shape(epoch_features)[-1]
        epoch_features = tf.reshape(epoch_features, [B, E, D])  # [B, E, D]

        pooled, attn_weights = self.pool(epoch_features, mask=attention_mask, return_weights=True)

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
    def __init__(self,
                 eegnet_args,
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
        self.eegnet = EEGNet(**eegnet_args)
        pooling_args = pooling_args or {}

        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])  # [B, C, T, 1] for EEGNet
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]
        
        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        self.classifier = layers.Dense(num_classes)  # No activation

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

class EEGNetFlatClassifierMod1(Model):
    def __init__(self,
                 eegnet_args,
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
        self.eegnet = EEGNetMod1(**eegnet_args)
        pooling_args = pooling_args or {}

        dummy_input = tf.zeros([1, eegnet_args['num_channels'], eegnet_args['num_samples'], 1])  # [B, C, T, 1] for EEGNet
        feature_dim = self.eegnet(dummy_input, training=False).shape[-1]
        
        self.pool = AttentionPooling(input_dim=feature_dim, **pooling_args)
        self.classifier = layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.l2(1e-3))  # No activation

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
