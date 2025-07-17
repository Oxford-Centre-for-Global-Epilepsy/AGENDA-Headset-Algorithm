import tensorflow as tf
from tensorflow.keras import regularizers, layers

class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim = 64, activation=tf.nn.tanh):
        super().__init__()
        self.scorer = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation=activation, kernel_initializer='glorot_uniform'),
            layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

    def call(self, x, mask=None, return_weights=False):
        """
        x: [B, E, D]
        mask: [B, E]
        return_weights: whether to return attention weights along with pooled output
        """
        # Attention-based pooling
        scores = self.scorer(x)  # [B, E, 1]
        scores = tf.squeeze(scores, axis=-1)  # [B, E]

        if mask is not None:
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

        weights = tf.nn.softmax(scores, axis=1)  # [B, E]
        weights_exp = tf.expand_dims(weights, axis=-1)  # [B, E, 1]
        pooled = tf.reduce_sum(x * weights_exp, axis=1)  # [B, D]

        if return_weights:
            return pooled, weights
        return pooled
