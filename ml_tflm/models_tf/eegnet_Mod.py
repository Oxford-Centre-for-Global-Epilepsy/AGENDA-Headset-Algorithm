import tensorflow as tf
from tensorflow.keras import layers, models

class EEGNetMod1(tf.keras.Model):
    """
    EEGNet with improved temporal kernel structure for IED detection,
    with optional MLP bottleneck and LayerNorm replacing BatchNorm for small EEG datasets.
    """
    def __init__(self,
                 num_channels=21,
                 num_samples=128,
                 dropout_rate=0.5,
                 F1=8,
                 D=2,
                 F2=16,
                 bottleneck_dim=None,
                 activation=tf.nn.elu):
        super().__init__()
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim

        # Temporal convolution: stacked 1×8 convs with dilation
        self.firstconv = models.Sequential([
            layers.Conv2D(F1, (1, 8), padding='same', dilation_rate=1, use_bias=False),
            layers.Conv2D(F1, (1, 8), padding='same', dilation_rate=2, use_bias=False),
            layers.LayerNormalization(axis=-1)
        ])

        # Depthwise spatial convolution
        self.depthwiseConv = models.Sequential([
            layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False),
            layers.LayerNormalization(axis=-1),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 4)),
            layers.Dropout(dropout_rate)
        ])

        # Separable convolution
        self.separableConv = models.Sequential([
            layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False),
            layers.LayerNormalization(axis=-1),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 8)),
            layers.Dropout(dropout_rate)
        ])

        # Optional bottleneck MLP layer
        if bottleneck_dim and bottleneck_dim > 0:
            self.bottleneck = models.Sequential([
                layers.Dense(bottleneck_dim, activation=self.activation),
                layers.Dropout(dropout_rate)
            ])
        else:
            self.bottleneck = None

    def call(self, inputs, return_features=False):
        """
        Parameters
        ----------
        inputs : Tensor
            Shape: [batch, channels, time, 1]
        return_features : bool
            If True, returns intermediate feature maps.
        """
        x = self.firstconv(inputs)                    # [B, 21, 128, F1]
        x1 = self.depthwiseConv(x)                    # [B, 1, 32, F1*D]
        x2 = self.separableConv(x1)                   # [B, 1, 4, F2]
        x_flat = tf.reshape(x2, [tf.shape(x2)[0], -1])  # [B, 64]

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
