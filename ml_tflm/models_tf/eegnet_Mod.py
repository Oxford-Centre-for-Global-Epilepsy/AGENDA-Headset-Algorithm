import tensorflow as tf
from tensorflow.keras import layers, models


class SplitConvStack(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(1, 8), use_bias=False,
                 activation=None, use_nonlinearity=True):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation
        self.use_nonlinearity = use_nonlinearity

        self.conv1_layers = [layers.Conv2D(1, kernel_size, padding='same', dilation_rate=1, use_bias=use_bias)
                             for _ in range(filters)]
        self.conv2_layers = [layers.Conv2D(1, kernel_size, padding='same', dilation_rate=2, use_bias=use_bias)
                             for _ in range(filters)]
        self.conv3_layers = [layers.Conv2D(1, kernel_size, padding='same', dilation_rate=4, use_bias=use_bias)
                             for _ in range(filters)]

    def call(self, x):
        outputs = []
        for i in range(self.filters):
            xi = x[:, :, :, i:i+1]  # [B, C, T, 1]
            y = self.conv1_layers[i](xi)
            if self.use_nonlinearity and self.activation:
                y = self.activation(y)
            y = self.conv2_layers[i](y)
            if self.use_nonlinearity and self.activation:
                y = self.activation(y)
            y = self.conv3_layers[i](y)
            if self.use_nonlinearity and self.activation:
                y = self.activation(y)
            outputs.append(y)
        return tf.concat(outputs, axis=-1)


class EEGNetMod1(tf.keras.Model):
    def __init__(self,
                 num_channels=21,
                 num_samples=128,
                 dropout_rate=0.5,
                 F1=8,
                 D=2,
                 F2=16,
                 bottleneck_dim=None,
                 activation=tf.nn.elu,
                 use_nonlinearity=True):
        super().__init__()
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim
        self.use_nonlinearity = use_nonlinearity

        self.temporal_project = layers.Conv2D(F1, (1, 1), padding='same', use_bias=False)

        self.firstconv = models.Sequential([
            self.temporal_project,
            SplitConvStack(F1, activation=self.activation, use_nonlinearity=use_nonlinearity),
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
