import tensorflow as tf
from tensorflow.keras import layers

class LSTMPooling(tf.keras.layers.Layer):
    """
    LSTM pooling layer that compresses a temporal sequence [B, E, D]
    into a single representation [B, H] using the final LSTM hidden state.

    Args:
        input_dim (int): Input feature dimension D (not used, kept for interface consistency).
        lstm_units (int): Number of LSTM units.
        bidirectional (bool): Use bidirectional LSTM.
    """
    def __init__(self, input_dim, lstm_units=32, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_heads = 1

        if bidirectional:
            self.lstm = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(lstm_units, return_sequences=True)
            )
            self.output_dim = lstm_units * 2
        else:
            self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True)
            self.output_dim = lstm_units

    def call(self, x, mask=None, return_weights=False):
        """
        Args:
            x: Tensor of shape [B, E, D]
            mask: Optional boolean mask of shape [B, E]
            return_weights: If True, returns dummy uniform weights of shape [B, 1, E]

        Returns:
            pooled: [B, H] or [B, 2H] if bidirectional
            dummy_weights (optional): [B, 1, E] uniform weights
        """
        lstm_out = self.lstm(x, mask=mask)  # [B, E, H]

        if mask is not None:
            # Expand mask: [B, E] -> [B, E, 1]
            mask_exp = tf.cast(mask[..., tf.newaxis], dtype=lstm_out.dtype)
            masked_out = lstm_out * mask_exp
            sum_mask = tf.reduce_sum(mask_exp, axis=1)  # [B, 1]
            pooled = tf.reduce_sum(masked_out, axis=1) / tf.maximum(sum_mask, 1e-8)  # [B, H]
        else:
            pooled = tf.reduce_mean(lstm_out, axis=1)  # [B, H]

        if return_weights:
            B, E = tf.shape(x)[0], tf.shape(x)[1]
            dummy_weights = tf.ones([B, 1, E], dtype=x.dtype) / tf.cast(E, x.dtype)
            return pooled, dummy_weights

        return pooled

class GRUPooling(tf.keras.layers.Layer):
    """
    GRU pooling layer that compresses a temporal sequence [B, E, D]
    into a single representation [B, H] using the final GRU hidden state.

    Args:
        input_dim (int): Input feature dimension D (not used, kept for interface consistency).
        gru_units (int): Number of GRU units.
        bidirectional (bool): Use bidirectional GRU.
    """
    def __init__(self, input_dim, gru_units=32, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_heads = 1

        if bidirectional:
            self.gru = tf.keras.layers.Bidirectional(
                tf.keras.layers.GRU(gru_units, return_sequences=True)
            )
            self.output_dim = gru_units * 2
        else:
            self.gru = tf.keras.layers.GRU(gru_units, return_sequences=True)
            self.output_dim = gru_units

    def call(self, x, mask=None, return_weights=False):
        """
        Args:
            x: Tensor of shape [B, E, D]
            mask: Optional boolean mask of shape [B, E]
            return_weights: If True, returns dummy uniform weights of shape [B, 1, E]

        Returns:
            pooled: [B, H] or [B, 2H] if bidirectional
            dummy_weights (optional): [B, 1, E] uniform weights
        """
        gru_out = self.gru(x, mask=mask)  # [B, E, H]

        if mask is not None:
            # Expand mask: [B, E] -> [B, E, 1]
            mask_exp = tf.cast(mask[..., tf.newaxis], dtype=gru_out.dtype)
            masked_out = gru_out * mask_exp
            sum_mask = tf.reduce_sum(mask_exp, axis=1)  # [B, 1]
            pooled = tf.reduce_sum(masked_out, axis=1) / tf.maximum(sum_mask, 1e-8)  # [B, H]
        else:
            pooled = tf.reduce_mean(gru_out, axis=1)  # [B, H]

        if return_weights:
            B, E = tf.shape(x)[0], tf.shape(x)[1]
            dummy_weights = tf.ones([B, 1, E], dtype=x.dtype) / tf.cast(E, x.dtype)
            return pooled, dummy_weights

        return pooled

class ConformerBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, ff_dim, num_heads, conv_kernel_size=31, dropout=0.5):
        super().__init__()
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.ffn1 = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='swish'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(input_dim),
            tf.keras.layers.Dropout(dropout),
        ])

        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.mhsa = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim//num_heads, dropout=dropout)

        self.layernorm3 = tf.keras.layers.LayerNormalization()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=2*input_dim, kernel_size=1, padding='same', activation='swish'),
            tf.keras.layers.DepthwiseConv1D(kernel_size=conv_kernel_size, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv1D(filters=input_dim, kernel_size=1, padding='same'),
        ])

        self.layernorm4 = tf.keras.layers.LayerNormalization()
        self.ffn2 = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='swish'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(input_dim),
            tf.keras.layers.Dropout(dropout),
        ])

    def call(self, x, mask=None):
        # FFN 1
        x1 = self.layernorm1(x)
        x = x + 0.5 * self.ffn1(x1)

        # MHSA
        x2 = self.layernorm2(x)
        attn_mask = tf.cast(mask[:, tf.newaxis, :], tf.bool) if mask is not None else None
        x = x + self.mhsa(x2, x2, attention_mask=attn_mask)

        # Conv
        x3 = self.layernorm3(x)
        x_conv = self.conv(x3)
        x = x + x_conv

        # FFN 2
        x4 = self.layernorm4(x)
        x = x + 0.5 * self.ffn2(x4)

        return x

class ConformerPooling(tf.keras.layers.Layer):
    """
    Conformer pooling layer with mean pooling over time.
    Args:
        input_dim (int): Input feature dimension D.
        ff_dim (int): Intermediate FFN dimension.
        num_heads (int): Number of MHSA heads.
        conv_kernel_size (int): Kernel size of depthwise convolution.
        num_blocks (int): Number of stacked Conformer blocks.
    """
    def __init__(self, input_dim, ff_dim=128, num_heads=4, conv_kernel_size=31, num_blocks=1, dropout=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = input_dim  # Since residual keeps dimensions

        self.blocks = [
            ConformerBlock(input_dim, ff_dim, num_heads, conv_kernel_size=conv_kernel_size, dropout=dropout)
            for _ in range(num_blocks)
        ]

    def call(self, x, mask=None, return_weights=False):
        """
        Args:
            x: Tensor of shape [B, E, D]
            mask: Optional boolean mask [B, E]
            return_weights: If True, returns dummy uniform weights [B, 1, E]
        Returns:
            pooled: [B, D]
            dummy_weights (optional): [B, 1, E]
        """
        for block in self.blocks:
            x = block(x, mask=mask)

        if mask is not None:
            mask_exp = tf.cast(mask[..., tf.newaxis], dtype=x.dtype)  # [B, E, 1]
            sum_mask = tf.reduce_sum(mask_exp, axis=1)  # [B, 1]
            pooled = tf.reduce_sum(x * mask_exp, axis=1) / tf.maximum(sum_mask, 1e-8)  # [B, D]
        else:
            pooled = tf.reduce_mean(x, axis=1)

        if return_weights:
            B, E = tf.shape(x)[0], tf.shape(x)[1]
            dummy_weights = tf.ones([B, 1, E], dtype=x.dtype) / tf.cast(E, x.dtype)
            return pooled, dummy_weights

        return pooled

class TransformerCLSPooling(tf.keras.layers.Layer):
    """
    Lightweight Transformer pooling using a CLS token.

    Args:
        input_dim (int): Dimensionality of input features D.
        ff_dim (int): Feedforward hidden dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        num_blocks (int): Number of Transformer blocks.
    """

    def __init__(self, input_dim, ff_dim=32, num_heads=1, dropout=0.1, num_blocks=1):
        super().__init__()
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.input_dim = input_dim

        self.cls_token = self.add_weight(
            shape=(1, 1, input_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token"
        )

        self.encoders = []
        for _ in range(num_blocks):
            self.encoders.append([
                layers.LayerNormalization(),
                layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim),
                layers.Dropout(dropout),
                layers.LayerNormalization(),
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(input_dim),
                layers.Dropout(dropout)
            ])

    def call(self, x, mask=None, return_weights=False):
        """
        Args:
            x: [B, E, D] sequence of embeddings
            mask: Optional boolean mask [B, E]
            return_weights: If True, return attention weights from CLS token [B, H, E]

        Returns:
            pooled: [B, D]
            cls_attn_weights (optional): [B, H, E] attention from CLS token to each timestep
        """
        B = tf.shape(x)[0]

        # Insert CLS token at the beginning
        cls = tf.tile(self.cls_token, [B, 1, 1])  # [B, 1, D]
        x = tf.concat([cls, x], axis=1)  # [B, E+1, D]

        if mask is not None:
            cls_mask = tf.ones((B, 1), dtype=mask.dtype)
            mask = tf.concat([cls_mask, tf.cast(mask, mask.dtype)], axis=1)  # [B, E+1]

        attention_mask = mask[:, tf.newaxis, tf.newaxis, :] if mask is not None else None

        cls_attn_weights = None

        for i, (ln1, mha, drop1, ln2, dense1, dense2, drop2) in enumerate(self.encoders):
            normed_x = ln1(x)
            attn_out, attn_weights = mha(
                query=normed_x,
                value=x,
                key=x,
                attention_mask=attention_mask,
                return_attention_scores=True
            )
            x = x + drop1(attn_out)

            if i == 0 and return_weights:
                cls_attn_weights = attn_weights[:, :, 0, 1:]  # [B, H, E]

            ff_out = dense2(dense1(ln2(x)))
            x = x + drop2(ff_out)

        cls_pooled = x[:, 0, :]  # [B, D]

        if return_weights:
            return cls_pooled, cls_attn_weights

        return cls_pooled
