import tensorflow as tf
from tensorflow.keras import layers

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
        self.dense1 = layers.Dense(hidden_dim)
        self.dense2 = layers.Dense(1)

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
