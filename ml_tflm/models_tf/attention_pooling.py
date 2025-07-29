import tensorflow as tf
from tensorflow.keras import layers

class MultiHeadAttentionPooling(tf.keras.layers.Layer):
    """
    Multi-head attention pooling layer using registered scorer name.

    Args:
        scorer (str): Name of the scorer defined in SCORER_REGISTRY.
        input_dim (int): Input feature dimension D.
        num_heads (int): Number of attention heads.
        **scorer_kwargs: Additional kwargs passed to scorer constructor.
    """
    def __init__(self, scorer, input_dim, num_heads=1, **scorer_kwargs):
        super().__init__()
        self.num_heads = num_heads

        if scorer not in SCORER_REGISTRY:
            raise ValueError(f"Unknown scorer '{scorer}'. Available options: {list(SCORER_REGISTRY.keys())}")

        scorer_cls = SCORER_REGISTRY[scorer]
        self.scorers = [
            scorer_cls(input_dim=input_dim, **scorer_kwargs)
            for _ in range(num_heads)
        ]

    def call(self, x, mask=None, return_weights=False):
        pooled_outputs = []
        all_weights = []

        for scorer in self.scorers:
            scores = scorer(x)  # [B, E, 1]
            scores = tf.squeeze(scores, axis=-1)  # [B, E]

            if mask is not None:
                scores = tf.where(mask, scores, tf.fill(tf.shape(scores), -1e9))

            weights = tf.nn.softmax(scores, axis=1)  # [B, E]
            pooled = tf.reduce_sum(x * tf.expand_dims(weights, axis=-1), axis=1)  # [B, D]

            pooled_outputs.append(pooled)
            all_weights.append(weights)

        pooled_concat = tf.concat(pooled_outputs, axis=-1) if self.num_heads > 1 else pooled_outputs[0]

        if return_weights:
            weights_stacked = tf.stack(all_weights, axis=1) if self.num_heads > 1 else tf.expand_dims(all_weights[0], axis=1)
            return pooled_concat, weights_stacked

        return pooled_concat

class MLPScorer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=16, activation=tf.nn.tanh):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation=activation, kernel_initializer='glorot_uniform'),
            layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

    def call(self, x):
        return self.mlp(x)

class GatedMLPScorer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=64):
        """
        Args:
            input_dim (int): Dimensionality of each segment vector.
            hidden_dim (int): Size of hidden layer for both gate and content projection.
        """
        super().__init__()
        self.content_proj = tf.keras.layers.Dense(hidden_dim, activation='tanh')     # V
        self.gate_proj = tf.keras.layers.Dense(hidden_dim, activation='sigmoid')     # U
        self.score_proj = tf.keras.layers.Dense(1, use_bias=False)                   # w

    def call(self, x):
        """
        Args:
            x: Tensor of shape [B, E, D]
        Returns:
            scores: Tensor of shape [B, E, 1]
        """
        gated = self.content_proj(x) * self.gate_proj(x)   # [B, E, H]
        scores = self.score_proj(gated)                    # [B, E, 1]
        return scores

class LSTMScorer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim=8, dropout=0.0):
        """
        Args:
            input_dim (int): Feature dimension of input.
            lstm_units (int): Number of units in the LSTM.
            dropout (float): Dropout rate inside LSTM (optional).
        """
        super().__init__()
        self.lstm = layers.Bidirectional(
            layers.LSTM(hidden_dim, return_sequences=True, dropout=dropout)
        )
        self.score_layer = layers.Dense(1, kernel_initializer='glorot_uniform')

    def call(self, x):
        """
        x: [B, E, D]
        Returns:
            scores: [B, E, 1]
        """
        lstm_out = self.lstm(x)  # [B, E, 2 * lstm_units]
        scores = self.score_layer(lstm_out)  # [B, E, 1]
        return scores

class TCNHybridScorer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        hidden_dim=32,
        kernel_size=3,
        num_layers=3,
        dilation_base=2,
        projection_dim=32,
        dropout=0.1
    ):
        """
        Args:
            input_dim (int): Dimension of each segment (e.g., 16)
            hidden_dim (int): Number of filters in each TCN layer
            kernel_size (int): TCN convolution width
            num_layers (int): Number of TCN layers
            dilation_base (int): Dilation growth factor per layer
            projection_dim (int): Hidden layer size in MLP scorer
            dropout_rate (float): Dropout rate inside MLP scorer
        """
        super().__init__()

        self.tcn_layers = []
        dilation = 1
        for _ in range(num_layers):
            self.tcn_layers.append(
                layers.SeparableConv1D(
                    filters=hidden_dim,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    padding='same',
                    activation='relu',
                    depthwise_initializer='glorot_uniform',
                    pointwise_initializer='glorot_uniform'
                )
            )
            dilation *= dilation_base

        self.score_mlp = tf.keras.Sequential([
            layers.Dense(projection_dim, activation='relu'),
            layers.Dropout(dropout), 
            layers.Dense(1, kernel_initializer='glorot_uniform')
        ])

    def call(self, x, training=None):
        """
        Args:
            x: Tensor of shape [B, E, D]
            training (bool): Whether in training mode (controls dropout)
        Returns:
            scores: Tensor of shape [B, E, 1]
        """
        h = x
        for conv in self.tcn_layers:
            h = conv(h)  # [B, E, hidden_dim]

        combined = tf.concat([x, h], axis=-1)  # [B, E, D + hidden_dim]
        scores = self.score_mlp(combined, training=training)  # [B, E, 1]
        return scores

class GatedTCNScorer(tf.keras.layers.Layer):
    def __init__(self, input_dim, tcn_hidden=32, mlp_dim=64, kernel_size=3, num_layers=3, dilation_base=2):
        super().__init__()

        # Local content projection (pure MLP)
        self.content_proj = tf.keras.layers.Dense(mlp_dim, activation='tanh')

        # TCN for context-aware gate
        self.tcn_layers = []
        dilation = 1
        for _ in range(num_layers):
            self.tcn_layers.append(
                tf.keras.layers.SeparableConv1D(
                    filters=tcn_hidden,
                    kernel_size=kernel_size,
                    dilation_rate=dilation,
                    padding='same',
                    activation='relu'
                )
            )
            dilation *= dilation_base
        self.gate_proj = tf.keras.layers.Dense(mlp_dim, activation='sigmoid')

        # Final scorer
        self.score_proj = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, x):
        # Content: local
        content = self.content_proj(x)  # [B, E, mlp_dim]

        # Gate: contextual
        h = x
        for conv in self.tcn_layers:
            h = conv(h)  # [B, E, tcn_hidden]
        gate = self.gate_proj(h)        # [B, E, mlp_dim]

        gated = content * gate          # [B, E, mlp_dim]
        scores = self.score_proj(gated) # [B, E, 1]
        return scores

class AVGScorer(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

    def call(self, x):
        """
        Args:
            x: Tensor of shape [B, E, D]
        Returns:
            scores: Tensor of shape [B, E, 1] — all zeros
        """
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]
        return tf.zeros([batch_size, seq_len, 1], dtype=x.dtype)

class AblationScorer(tf.keras.layers.Layer):
    def __init__(self, input_dim, keep_ratio=1.0, seed=None):
        """
        Args:
            input_dim (int): Unused, included for registry compatibility.
            keep_ratio (float): Proportion of segments to retain (0 < keep_ratio <= 1.0).
            seed (int or None): Optional random seed for reproducibility.
        """
        super().__init__()
        self.keep_ratio = keep_ratio
        self.seed = seed

    def call(self, x, training=None):
        B, E = tf.shape(x)[0], tf.shape(x)[1]
        dtype = x.dtype

        k = tf.cast(tf.math.round(self.keep_ratio * tf.cast(E, tf.float32)), tf.int32)

        if self.seed is not None:
            tf.random.set_seed(self.seed)

        # Random scores per segment per batch
        random_vals = tf.random.uniform([B, E], seed=self.seed)
        topk_vals, topk_indices = tf.math.top_k(random_vals, k=k)  # [B, k]

        # Build a mask: True for kept, False otherwise
        batch_indices = tf.repeat(tf.range(B), repeats=k)  # [B * k]
        segment_indices = tf.reshape(topk_indices, [-1])   # [B * k]
        full_indices = tf.stack([batch_indices, segment_indices], axis=1)  # [B*k, 2]

        flat_mask = tf.scatter_nd(
            indices=full_indices,
            updates=tf.ones([B * k], dtype=tf.bool),
            shape=[B, E]
        )

        keep_mask = tf.expand_dims(flat_mask, axis=-1)  # [B, E, 1]
        scores = tf.where(keep_mask, tf.zeros_like(x[..., :1]), tf.fill([B, E, 1], float("-inf")))

        return scores


SCORER_REGISTRY = {
    "MLP": MLPScorer,
    "MLPGated": GatedMLPScorer,
    "LSTM": LSTMScorer,
    "TCNHybrid": TCNHybridScorer,
    "TCNGated": GatedTCNScorer,
    "AVG": AVGScorer,
    "Ablation": AblationScorer
    }

def GetPoolingLayer(pool='ATTN', **pool_kwargs):
    return POOLING_REGISTRY[pool](**pool_kwargs)

from ml_tflm.models_tf.experimental_pooling import LSTMPooling, GRUPooling

POOLING_REGISTRY = {
    "ATTN": MultiHeadAttentionPooling,
    "LSTM": LSTMPooling,
    "GRU":GRUPooling
}

if __name__ == "__main__":
    # Create dummy input: batch of 3 sequences, each with 10 time steps and 32-dim features
    batch_size = 3
    sequence_len = 10
    feature_dim = 32
    dummy_input = tf.random.normal([batch_size, sequence_len, feature_dim])

    # Optional: create a mask (e.g., only first 7 time steps are valid)
    mask = tf.sequence_mask([7, 5, 9], maxlen=sequence_len)  # shape [3, 10]

    print("\n=== MultiHeadAttentionPooling with 1 head (single-head case) ===")
    single_head_pool = MultiHeadAttentionPooling(input_dim=feature_dim, scorer="MLPScorer", num_heads=1)
    pooled_single, weights_single = single_head_pool(dummy_input, mask=mask, return_weights=True)
    print("Pooled output shape:", pooled_single.shape)       # Expected: [3, 32]
    print("Attention weights shape:", weights_single.shape)  # Expected: [3, 1, 10]

    print("\n=== MultiHeadAttentionPooling with 4 heads ===")
    multi_head_pool = MultiHeadAttentionPooling(input_dim=feature_dim, scorer="MLPScorer", num_heads=4)
    pooled_multi, weights_multi = multi_head_pool(dummy_input, mask=mask, return_weights=True)
    print("Pooled output shape:", pooled_multi.shape)       # Expected: [3, 128] (32 * 4)
    print("Attention weights shape:", weights_multi.shape)  # Expected: [3, 4, 10]
