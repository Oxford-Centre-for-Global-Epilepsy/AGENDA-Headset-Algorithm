import tensorflow as tf
from tensorflow.keras import layers, models

class EEGNet(tf.keras.Model):
    """
    TensorFlow EEGNet implementation with LayerNormalization for per-epoch EEG feature extraction.
    """

    def __init__(self,
                 num_channels=21,
                 num_samples=128,
                 dropout_rate=0.5,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_length=64,
                 bottleneck_dim=None,
                 activation=tf.nn.elu):

        super().__init__()
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim

        self.firstconv = models.Sequential([
            layers.Conv2D(F1, (1, kernel_length), padding='same', use_bias=False),
            layers.LayerNormalization()
        ])

        self.depthwiseConv = models.Sequential([
            layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False),
            layers.LayerNormalization(),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 4)),
            layers.Dropout(dropout_rate)
        ])

        self.separableConv = models.Sequential([
            layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False),
            layers.LayerNormalization(),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 8)),
            layers.Dropout(dropout_rate)
        ])

        # Optional bottleneck
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

        Parameters
        ----------
        inputs : Tensor
            Shape: [batch, channels, time, 1]
        return_features : bool
            If True, returns intermediate feature maps.

        Returns
        -------
        Tensor or dict
            Feature vector or dictionary of intermediate outputs.
        """
        x = self.firstconv(inputs)
        x1 = self.depthwiseConv(x)
        x2 = self.separableConv(x1)
        x_flat = tf.reshape(x2, [tf.shape(x2)[0], -1])

        if self.bottleneck is not None:
            x_flat = self.bottleneck(x_flat)

        if return_features:
            return {
                "features": x_flat,
                "temporal_features": x,
                "spatial_features": x1,
                "final_features": x2
            }

        return x_flat
