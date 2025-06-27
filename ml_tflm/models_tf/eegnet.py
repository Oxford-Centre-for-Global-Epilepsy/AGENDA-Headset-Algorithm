import tensorflow as tf
from tensorflow.keras import layers, models

class EEGNet(tf.keras.Model):
    """
    TensorFlow EEGNet implementation for per-epoch EEG feature extraction.

    Parameters
    ----------
    num_channels : int
        Number of EEG channels.
    num_samples : int
        Number of time samples per epoch.
    dropout_rate : float
        Dropout rate after conv layers.
    F1 : int
        Number of temporal filters.
    D : int
        Depth multiplier (spatial filters per temporal filter).
    F2 : int
        Number of pointwise filters.
    kernel_length : int
        Length of temporal kernel.
    activation : callable
        Activation function (e.g., tf.nn.elu, tf.nn.relu6). Default is tf.nn.elu.
    """
    def __init__(self,
                 num_channels=21,
                 num_samples=256,
                 dropout_rate=0.5,
                 F1=8,
                 D=2,
                 F2=16,
                 kernel_length=64,
                 activation=tf.nn.elu):

        super().__init__()
        self.activation = activation

        # First temporal conv
        self.firstconv = models.Sequential([
            layers.Conv2D(F1, (1, kernel_length), padding='same', use_bias=False),
            layers.BatchNormalization()
        ])

        # Depthwise spatial conv
        self.depthwiseConv = models.Sequential([
            layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 4)),
            layers.Dropout(dropout_rate)
        ])

        # Separable conv
        self.separableConv = models.Sequential([
            layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 8)),
            layers.Dropout(dropout_rate)
        ])

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

        if return_features:
            return {
                "features": x_flat,
                "temporal_features": x,
                "spatial_features": x1,
                "final_features": x2
            }

        return x_flat
