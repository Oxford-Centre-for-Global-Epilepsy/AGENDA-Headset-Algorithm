import tensorflow as tf

class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=4, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels % self.groups != 0:
            raise ValueError(f"Channels ({channels}) must be divisible by groups ({self.groups})")
        self.gamma = self.add_weight(name="gamma", shape=(channels,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(channels,), initializer="zeros", trainable=True)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        G = self.groups
        x = tf.reshape(inputs, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, axes=[1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, H, W, C])
        return self.gamma * x + self.beta
