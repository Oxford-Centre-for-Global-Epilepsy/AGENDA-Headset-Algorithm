import tensorflow as tf
from tensorflow.keras import layers, models
from ml_tflm.models_tf.group_norm import GroupNormalization

class EEGNetMod1(tf.keras.Model):
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

        self.firstconv = models.Sequential([
            layers.Conv2D(F1, (1, 8), padding='same', dilation_rate=1, use_bias=False),
            layers.Conv2D(F1, (1, 8), padding='same', dilation_rate=2, use_bias=False),
            layers.LayerNormalization(axis=-1)
        ])

        self.depthwiseConv = models.Sequential([
            layers.DepthwiseConv2D((num_channels, 1), depth_multiplier=D, use_bias=False),
            layers.LayerNormalization(axis=-1),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 4)),
            layers.Dropout(dropout_rate)
        ])

        self.separableConv = models.Sequential([
            layers.SeparableConv2D(F2, (1, 16), padding='same', use_bias=False),
            layers.LayerNormalization(axis=-1),
            layers.Activation(self.activation),
            layers.AveragePooling2D((1, 8)),
            layers.Dropout(dropout_rate)
        ])

        if bottleneck_dim and bottleneck_dim > 0:
            self.bottleneck = models.Sequential([
                layers.Dense(bottleneck_dim, activation=self.activation),
                layers.Dropout(dropout_rate)
            ])
        else:
            self.bottleneck = None

    def call(self, inputs, return_features=False):
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