import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    """
    Multi-head attention pooling layer.

    Each head uses a separate attention scorer to extract different temporal patterns.
    If num_heads = 1, this behaves like single-head attention pooling.

    Input: 
        x    -- [B, E, D] (sequence of embeddings)
        mask -- [B, E] (optional boolean mask)
    Output:
        pooled: [B, D * num_heads] if num_heads > 1 else [B, D]
        (optional) weights: [B, num_heads, E]
    """

    def __init__(self, input_dim, num_heads=1, hidden_dim=16, activation=tf.nn.tanh):
        """
        Args:
            input_dim (int): Dimensionality of input features D.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Hidden dimension of the attention MLP.
            activation (callable): Activation function for scorer MLP.
        """
        super().__init__()
        self.num_heads = num_heads

        # One scorer (MLP) per head
        self.scorers = [
            tf.keras.Sequential([
                layers.Dense(hidden_dim, activation=activation, kernel_initializer='glorot_uniform'),
                layers.Dense(1, kernel_initializer='glorot_uniform')
            ])
            for _ in range(num_heads)
        ]

    def call(self, x, mask=None, return_weights=False):
        """
        Args:
            x: Tensor of shape [B, E, D]
            mask: Optional mask of shape [B, E] with True for valid tokens
            return_weights: If True, also return attention weights [B, H, E]

        Returns:
            pooled: [B, D * H] or [B, D] if H=1
            (optional) attention weights: [B, H, E]
        """
        pooled_outputs = []
        all_weights = []

        for scorer in self.scorers:
            scores = scorer(x)  # [B, E, 1]
            scores = tf.squeeze(scores, axis=-1)  # [B, E]

            if mask is not None:
                scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))  # Masked positions -> large negative

            weights = tf.nn.softmax(scores, axis=1)  # [B, E]
            pooled = tf.reduce_sum(x * tf.expand_dims(weights, axis=-1), axis=1)  # [B, D]

            pooled_outputs.append(pooled)
            all_weights.append(weights)

        # Concatenate head outputs along feature axis
        pooled_concat = tf.concat(pooled_outputs, axis=-1) if self.num_heads > 1 else pooled_outputs[0]

        if return_weights:
            weights_stacked = tf.stack(all_weights, axis=1) if self.num_heads > 1 else tf.expand_dims(all_weights[0], axis=1)
            return pooled_concat, weights_stacked

        return pooled_concat

if __name__ == "__main__":
    # Create dummy input: batch of 3 sequences, each with 10 time steps and 32-dim features
    batch_size = 3
    sequence_len = 10
    feature_dim = 32
    dummy_input = tf.random.normal([batch_size, sequence_len, feature_dim])

    # Optional: create a mask (e.g., only first 7 time steps are valid)
    mask = tf.sequence_mask([7, 5, 9], maxlen=sequence_len)  # shape [3, 10]

    print("\n=== MultiHeadAttentionPooling with 1 head (single-head case) ===")
    single_head_pool = MultiHeadAttentionPooling(input_dim=feature_dim, num_heads=1)
    pooled_single, weights_single = single_head_pool(dummy_input, mask=mask, return_weights=True)
    print("Pooled output shape:", pooled_single.shape)       # Expected: [3, 32]
    print("Attention weights shape:", weights_single.shape)  # Expected: [3, 1, 10]

    print("\n=== MultiHeadAttentionPooling with 4 heads ===")
    multi_head_pool = MultiHeadAttentionPooling(input_dim=feature_dim, num_heads=4)
    pooled_multi, weights_multi = multi_head_pool(dummy_input, mask=mask, return_weights=True)
    print("Pooled output shape:", pooled_multi.shape)       # Expected: [3, 128] (32 * 4)
    print("Attention weights shape:", weights_multi.shape)  # Expected: [3, 4, 10]
