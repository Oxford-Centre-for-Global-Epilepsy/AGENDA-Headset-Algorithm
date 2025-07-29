import tensorflow as tf

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
