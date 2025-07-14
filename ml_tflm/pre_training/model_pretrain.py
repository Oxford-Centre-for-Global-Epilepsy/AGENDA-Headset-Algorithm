import tensorflow as tf

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})

from tensorflow.keras import layers

class L2Normalization(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape  # normalization doesn't change shape

    def get_config(self):
        config = super().get_config()
        config.update({'axis': self.axis})
        return config


def build_feature_extractor(num_channels=21,
                             num_samples=128,
                             dropout_rate=0.5,
                             F1=8,
                             D=2,
                             F2=16,
                             bottleneck_dim=None,
                             activation=tf.nn.elu):
    """
    Returns EEG feature extractor as a tf.keras.Sequential model.
    """

    # Multi-branch temporal conv block with dilation (in parallel)
    input_layer = tf.keras.Input(shape=(num_channels, num_samples, 1))

    temporal_branches = []
    for d_rate in [1, 2, 4]:
        temporal_branches.append(
            layers.Conv2D(F1, (1, 8), padding='same', dilation_rate=d_rate, use_bias=False)(input_layer)
        )
    x = layers.Concatenate(axis=-1)(temporal_branches)  # [B, 21, 128, F1*3]
    x = layers.LayerNormalization(axis=-1)(x)

    # Depthwise spatial conv
    x = layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False)(x)
    x = layers.LayerNormalization(axis=-1)(x)
    x = layers.Activation(activation)(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable convolution
    x = layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = layers.LayerNormalization(axis=-1)(x)
    x = layers.Activation(activation)(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Flatten
    x = layers.Flatten()(x)

    # Optional bottleneck
    if bottleneck_dim and bottleneck_dim > 0:
        x = layers.Dense(bottleneck_dim, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    x = L2Normalization(axis=1)(x)

    return tf.keras.Model(inputs=input_layer, outputs=x, name="EEGFeatureExtractor")

def build_vanilla_eegnet(num_channels=21,
                       num_samples=128,
                       dropout_rate=0.5,
                       F1=8,
                       D=2,
                       F2=16,
                       bottleneck_dim=None,
                       activation=tf.nn.elu):
    """
    Returns a vanilla EEGNet-style feature extractor (single-branch temporal convolution).
    """

    input_layer = tf.keras.Input(shape=(num_channels, num_samples, 1))  # [B, C, T, 1]

    # Temporal convolution
    x = layers.Conv2D(F1, (1, 64), padding='same', use_bias=False)(input_layer)
    x = layers.BatchNormalization()(x)

    # Depthwise spatial convolution
    x = layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Separable convolution
    x = layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Flatten
    x = layers.Flatten()(x)

    # Optional bottleneck
    if bottleneck_dim and bottleneck_dim > 0:
        x = layers.Dense(bottleneck_dim, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)

    x = L2Normalization(axis=1)(x)

    return tf.keras.Model(inputs=input_layer, outputs=x, name="VanillaEEGNet")

def build_projector(input_dim,
                    projection_dim=64,
                    hidden_dim=128,
                    activation='relu',
                    dropout_rate=0.2,
                    name='Projector'):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = layers.Dense(hidden_dim, activation=activation)(inputs)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(projection_dim)(x)
    outputs = L2Normalization(axis=1)(x)

    return tf.keras.Model(inputs, outputs, name=name)

def build_attention(input_dim,
                    hidden_dim=8,
                    dropout_rate=0.0,
                    activation='tanh',
                    name='Attention'):
    """
    Returns a sequential MLP attention scorer: [B, T, D] → [B, T]
    All weights are initialized to zero.
    """

    inputs = tf.keras.Input(shape=(None, input_dim))  # shape = [T, D]
    x = layers.Dense(hidden_dim,
                    activation=activation,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                    bias_initializer='zeros', 
                    kernel_regularizer=tf.keras.regularizers.l2(2e-5))(inputs)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1,
                    use_bias=False,
                    kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3),
                    kernel_regularizer=tf.keras.regularizers.l2(2e-5))(x)
    outputs = layers.Flatten()(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

def masked_softmax(logits, mask, axis=-1):
    mask = tf.cast(mask, tf.float32)
    masked_logits = tf.where(mask > 0, logits, tf.float32.min)
    weights = tf.nn.softmax(masked_logits, axis=axis)
    weights *= mask
    weights /= tf.reduce_sum(weights, axis=axis, keepdims=True) + 1e-8
    return weights

class AGENDAEncoder(tf.keras.Model):
    def __init__(self, feature_extractor, projector, attention, name="AGENDAEncoder", clip_attention = False, Temerature = 0.4):
        super().__init__(name=name)
        self.feature_extractor = feature_extractor  # Model: [C, T_s, 1] → [D_feat]
        self.projector = projector                  # Model: [D_feat] → [D_proj]
        self.attention = attention                  # Model: [T, D_proj] → [T]
        self.mode = None
        self.pooling = False
        self.clip_attention = clip_attention
        self.temperature = Temerature

    def set_mode(self, mode):
        assert mode in {"phase1", "phase2", "build", "attention_test"}
        self.mode = mode

        if mode == "phase1":
            self.feature_extractor.trainable = True
            self.projector.trainable = True
            self.attention.trainable = False
            self.pooling = False

        elif mode == "phase2":
            self.feature_extractor.trainable = False
            self.projector.trainable = False
            self.attention.trainable = True
            self.pooling = True

        elif mode == "build":
            self.feature_extractor.trainable = True
            self.projector.trainable = True
            self.attention.trainable = True
            self.pooling = True

        elif mode == "attention_test":
            self.feature_extractor.trainable = False
            self.projector.trainable = False
            self.attention.trainable = True
            self.pooling = True

    def call(self, x, attention_mask=None, return_attn_weights=True, return_features=True, training=False):
        """
        Args:
            x: phase1 → [B, C, T]; phase2 → [B, E, C, T]
            attention_mask: [B, E] in phase2
            return_attn_weights: bool, return attention weights if True
            return_features: bool, return pooled features if True

        Returns:
            dict with keys:
                - "z_proj" (always)
                - "attn_logits" and "attn_weights" (if mode ≠ phase1)
                - "z_pooled" (if pooling=True)
                - "logits" (if classifier exists)
        """
        if self.mode == "phase1":
            x_feat = self.feature_extractor(x, training=training)   # [B, D_feat]
            z_proj = self.projector(x_feat, training=training)      # [B, D_proj]
            attn_logits = self.attention(tf.expand_dims(z_proj, 1), training=False)  # [B, 1]
            attn_logits = tf.squeeze(attn_logits, axis=1)           # [B]
            attn_weights = tf.nn.softmax(attn_logits, axis=0)       # [B]
            return {
                "z_proj": z_proj,
                "attn_logits": attn_logits,
                "attn_weights": attn_weights
            }

        if self.mode == "attention_test":
            assert len(x.shape) == 3, f"Expected [B, E, D_proj], got {x.shape}"
            z_proj = x  # Directly use `x` as projected features
        else:
            # phase2 or build
            # x: [B, E, C, T]
            B, E = tf.shape(x)[0], tf.shape(x)[1]

            # Ensure x is 5D: [B, E, C, T, 1]
            if x.shape.rank == 4:
                x = tf.expand_dims(x, -1)  # [B, E, C, T] → [B, E, C, T, 1]
            elif x.shape.rank != 5:
                raise ValueError(f"Expected x to be 4D or 5D, got shape {x.shape}")

            s = tf.shape(x)
            x = tf.reshape(x, [s[0] * s[1], s[2], s[3], s[4]])  # [B*E, C, T, 1]


            x_feat = self.feature_extractor(x, training=training)       # [B*E, D_feat]
            z_proj = self.projector(x_feat, training=training)          # [B*E, D_proj]
            z_proj = tf.reshape(z_proj, [B, E, -1])                     # [B, E, D_proj]

        attn_logits = self.attention(z_proj, training=training)     # [B, E]
        if self.clip_attention:
            # Tanh-saturate and optionally scale to boost contrast
            attn_logits = tf.tanh(attn_logits) / self.temperature

        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, tf.float32)
            attn_logits += tf.math.log(attention_mask + 1e-8)  # masking trick for softmax
        attn_weights = tf.nn.softmax(attn_logits, axis=1)
        
        z_pooled = tf.reduce_sum(z_proj * tf.expand_dims(attn_weights, -1), axis=1)  # [B, D_proj]

        result = {
            "z_proj": z_proj,
            "attn_logits": attn_logits,
            "attn_weights": attn_weights,
            "z_pooled": z_pooled,
        }

        if not return_attn_weights:
            result.pop("attn_weights")
        if not return_features:
            result.pop("z_pooled")

        return result

def configure_biphasic_model(feature_args, projector_args, attention_args):
    """
    Creates a biphasic AGENDAEncoder model.

    Args:
        feature_args: dict for feature extractor builder
        projector_args: dict for projector builder
        attention_args: dict for attention builder

    Returns:
        AGENDAEncoder model, initialized in 'inference' mode
    """
    feature_extractor = build_vanilla_eegnet(**feature_args)
    projector = build_projector(**projector_args)
    attention = build_attention(**attention_args)

    model = AGENDAEncoder(
        feature_extractor=feature_extractor,
        projector=projector,
        attention=attention
    )

    model.set_mode("build")  # Safe default
    return model

if __name__ == "__main__":
    feature_args = {
        "bottleneck_dim": 16, 
    }

    projector_args = {
        "input_dim": 16,
        "projection_dim": 32
    }

    attention_args = {
        "input_dim": 32,
    }

    model = configure_biphasic_model(feature_args, projector_args, attention_args)

    
    # === Phase 1 Inference ===
    model.set_mode("phase1")
    B, C, T = 4, 21, 128
    input_phase1 = tf.random.normal((B, C, T))
    output_phase1 = model(input_phase1, training=False)

    print("=== Phase 1 Output ===")
    for k, v in output_phase1.items():
        print(f"{k}: {v.shape}")

    # === Phase 2 Inference ===
    model.set_mode("phase2")
    B, E, C, T = 2, 5, 21, 128
    input_phase2 = tf.random.normal((B, E, C, T))
    attention_mask = tf.constant([[1, 1, 1, 0, 0],
                                [1, 1, 0, 0, 0]], dtype=tf.int32)

    output_phase2 = model(input_phase2, attention_mask=attention_mask, return_features = True, training=False)

    print("\n=== Phase 2 Output ===")
    for k, v in output_phase2.items():
        print(f"{k}: {v.shape}")