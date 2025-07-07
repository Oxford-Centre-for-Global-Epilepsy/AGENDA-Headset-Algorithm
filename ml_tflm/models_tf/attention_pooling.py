import tensorflow as tf
from tensorflow.keras import regularizers

class AttentionPooling(tf.keras.layers.Layer):
    """
    Attention pooling using an intermediate latent head dimension for diverse feature detection.

    Parameters
    ----------
    input_dim : int
        Feature dimension per epoch (after EEGNet).
    num_heads : int
        Size of the latent attention head space (e.g., 4 or 8).
    activation : callable
        Activation function to use between layers.
    trainable_attention : bool
        Whether to use attention or fall back to average pooling.
    head_aggregation : str
        Method to combine head scores into final score. Options: 'sum', 'mean', 'mlp'.
    """
    def __init__(self, input_dim, num_heads=4, activation=tf.nn.tanh,
                 trainable_attention=False, head_aggregation='sum'):
        super().__init__()
        self.trainable_attention = trainable_attention
        self.head_aggregation = head_aggregation
        self.dense_heads = tf.keras.layers.Dense(
            num_heads,
            kernel_initializer="zeros",
            bias_initializer="zeros",
            kernel_regularizer=regularizers.l2(1e-4),  # L2 regularization
        )
        self.activation = activation

        if head_aggregation == 'mlp':
            self.head_combiner = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation=activation, kernel_initializer="zeros", bias_initializer="zeros"),
                tf.keras.layers.Dense(1, kernel_initializer="zeros", bias_initializer="zeros")
            ])

    def toggle_trainability(self, enable=True):
        """
        Turn on or off attention pooling.
        When off, the layer falls back to average pooling.
        """
        self.trainable_attention = enable

    def get_trainability(self):
        return self.trainable_attention

    def call(self, x, mask=None, return_weights=False):
        """
        x: [B, E, D]
        mask: [B, E], boolean
        return_weights: bool
        """
        if not self.trainable_attention:
            if mask is not None:
                mask = tf.cast(mask, x.dtype)
                mask_exp = tf.expand_dims(mask, axis=-1)
                pooled = tf.reduce_sum(x * mask_exp, axis=1) / tf.reduce_sum(mask_exp, axis=1)
            else:
                pooled = tf.reduce_mean(x, axis=1)
            if return_weights:
                B, E = tf.shape(x)[0], tf.shape(x)[1]
                weights = tf.ones([B, E], dtype=x.dtype) / tf.cast(E, x.dtype)
                return pooled, weights
            return pooled

        # Step 1: latent attention heads
        scores = self.activation(self.dense_heads(x))  # [B, E, H]

        # Step 2: aggregate heads → scalar score per epoch
        if self.head_aggregation == 'sum':
            scores = tf.reduce_sum(scores, axis=-1)  # [B, E]
        elif self.head_aggregation == 'mean':
            scores = tf.reduce_mean(scores, axis=-1)  # [B, E]
        elif self.head_aggregation == 'mlp':
            scores = self.head_combiner(scores)  # [B, E, 1]
            scores = tf.squeeze(scores, axis=-1)  # [B, E]
        else:
            raise ValueError(f"Unknown head aggregation method: {self.head_aggregation}")

        # Step 3: mask and softmax
        if mask is not None:
            scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))
        weights = tf.nn.softmax(scores, axis=1)  # [B, E]

        # Step 4: weighted sum over epochs
        weights_exp = tf.expand_dims(weights, axis=-1)  # [B, E, 1]
        pooled = tf.reduce_sum(x * weights_exp, axis=1)  # [B, D]

        if return_weights:
            return pooled, weights
        return pooled
    
    def build(self, input_shape):
        super().build(input_shape)

