import tensorflow as tf
from tensorflow.keras import layers, models

class EEGNet(tf.keras.Model):
    """
    TensorFlow EEGNet implementation with LayerNormalization for per-epoch EEG feature extraction.

    This architecture is inspired by EEGNet (Lawhern et al., 2018), designed for efficient EEG-based classification.
    It uses temporal, spatial, and separable convolutions to extract frequency and location-aware features from EEG signals.

    Inputs:
        [batch_size, num_channels, num_samples, 1]
    """

    def __init__(self,
                 num_channels=21,
                 num_samples=128,
                 dropout_rate=0.5,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_length=64,
                 temporal_type='vanilla',
                 bottleneck_dim=None,
                 activation=tf.nn.elu):
        """
        Initializes EEGNet model layers.

        Args:
            num_channels (int): Number of EEG channels.
            num_samples (int): Number of time samples per segment.
            dropout_rate (float): Dropout rate used after spatial and separable conv blocks.
            F1 (int): Number of filters for the first temporal convolution.
            D (int): Depth multiplier for depthwise convolution (spatial filters).
            F2 (int): Number of pointwise filters in separable convolution block.
            kernel_length (int): Length of the temporal convolution kernel.
            temporal_type (string): Name of the temporal convolution block type.
            bottleneck_dim (int or None): If set, adds a dense bottleneck layer of this size.
            activation (function): Activation function to use (e.g., tf.nn.elu).
        """
        super().__init__()
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim

        # Temporal convolution block (extracts frequency features)
        if temporal_type=='vanilla':
            self.firstconv = models.Sequential([
                layers.Conv2D(F1, (1, kernel_length), padding='same', use_bias=False),
                layers.LayerNormalization()
            ])
        elif temporal_type == 'multiscale':
            dilation_rates = [1, 2, 4, 8]
            filters_per_branch = max(F1 // len(dilation_rates), 2)

            self.firstconv = models.Sequential([
                # Each branch extracts F1 // 4 filters
                MultiScaleTemporalConv(filters_per_branch=filters_per_branch,
                                       kernel_size=(1, kernel_length),
                                       dilation_rates=dilation_rates,
                                       activation=activation),
                layers.LayerNormalization()
            ])
        else:
            raise ValueError(f"Unknown temporal_type: {temporal_type}")

        # Spatial filtering block using depthwise convolution
        self.depthwiseConv = models.Sequential([
            layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False),
            layers.LayerNormalization(),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 4)),  # Downsample time dimension
            layers.Dropout(dropout_rate)
        ])

        # Separable convolution block (pointwise conv after spatial)
        self.separableConv = models.Sequential([
            layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False),
            layers.LayerNormalization(),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 8)),  # Further downsampling
            layers.Dropout(dropout_rate)
        ])

        # Optional dense bottleneck layer to reduce feature dimensionality
        if bottleneck_dim and bottleneck_dim > 0:
            self.bottleneck = models.Sequential([
                layers.Dense(bottleneck_dim, activation=self.activation),
                layers.Dropout(dropout_rate)
            ])
        else:
            self.bottleneck = None

    def call(self, inputs, return_features=False):
        """
        Forward pass of EEGNet.

        Args:
            inputs (Tensor): EEG input with shape [batch, channels, time, 1]
            return_features (bool): If True, return intermediate feature maps in a dictionary.

        Returns:
            Tensor or dict:
                - If return_features=False: a [batch, feature_dim] tensor
                - If return_features=True: a dict with intermediate activations
        """
        # Apply temporal convolution
        x = self.firstconv(inputs)  # [B, C, T, F1]
        
        # Apply spatial filtering via depthwise conv
        x1 = self.depthwiseConv(x)  # [B, 1, T', F1 * D]
        
        # Apply separable convolution
        x2 = self.separableConv(x1)  # [B, 1, T'', F2]

        # Flatten spatial-temporal features
        x_flat = tf.reshape(x2, [tf.shape(x2)[0], -1])  # [B, feature_dim]

        # Optional bottleneck
        if self.bottleneck is not None:
            x_flat = self.bottleneck(x_flat)

        # Return intermediate layers for visualization or debugging
        if return_features:
            return {
                "features": x_flat,
                "temporal_features": x,   # after firstconv
                "spatial_features": x1,   # after depthwiseConv
                "final_features": x2      # after separableConv
            }

        # Default return: final feature vector
        return x_flat

class MultiScaleTemporalConv(tf.keras.layers.Layer):
    """
    Multi-scale temporal convolution block using parallel Conv2D layers
    with different dilation rates, without projection. This preserves
    filter-level interpretability, as in the original EEGNet.

    Input shape: [batch, channels, time, 1]
    Output shape: [batch, channels, time, total_filters] where
                  total_filters = filters_per_branch × num_dilations
    """

    def __init__(self, filters_per_branch=4, kernel_size=(1, 8), dilation_rates=(1, 2, 4, 8), 
                 activation=tf.nn.elu, use_nonlinearity=True):
        """
        Args:
            filters_per_branch (int): Number of filters for each dilation branch.
            kernel_size (tuple): Size of the temporal convolution kernel.
            dilation_rates (tuple): List of dilation rates to use in parallel.
            activation (callable): Activation function to apply after each branch.
            use_nonlinearity (bool): Whether to apply activation after each branch.
        """
        super().__init__()
        self.use_nonlinearity = use_nonlinearity
        self.activation = activation

        # Parallel temporal Conv2D layers with different dilation rates
        self.dilated_convs = [
            layers.Conv2D(filters=filters_per_branch,
                          kernel_size=kernel_size,
                          dilation_rate=d,
                          padding='same',
                          use_bias=False)
            for d in dilation_rates
        ]

        # Optional normalization for stability
        self.norm = layers.LayerNormalization()

    def call(self, x):
        # Apply all branches in parallel
        conv_outputs = []
        for conv in self.dilated_convs:
            y = conv(x)
            if self.use_nonlinearity and self.activation:
                y = self.activation(y)
            conv_outputs.append(y)

        # Concatenate outputs along the feature axis
        x_cat = tf.concat(conv_outputs, axis=-1)  # Total filters = filters_per_branch × num_dilations

        # Normalize concatenated output
        return self.norm(x_cat)

if __name__ == "__main__":
    # Define dummy input shape: batch of 2 samples, 21 EEG channels, 128 time points, 1 feature per channel
    dummy_input = tf.random.normal([2, 21, 128, 1])

    # Instantiate EEGNet with both 'vanilla' and 'multiscale' options
    print("\n=== EEGNet with Vanilla Temporal Convolution ===")
    model_vanilla = EEGNet(temporal_type='vanilla')
    output_vanilla = model_vanilla(dummy_input, return_features=True)
    model_vanilla.summary()

    print("\n=== EEGNet with Multiscale Temporal Convolution ===")
    model_multiscale = EEGNet(kernel_length=8, temporal_type='multiscale')
    output_multiscale = model_multiscale(dummy_input, return_features=True)
    model_multiscale.summary()
