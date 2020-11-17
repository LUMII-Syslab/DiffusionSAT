import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, axis=1, epsilon=1e-6, subtract_mean=False, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        num_units = input_shape.as_list()[-1]
        if self.subtract_mean:
            self.bias = self.add_weight("bias", [num_units], initializer=tf.zeros_initializer)

    def call(self, inputs, **kwargs):
        if self.subtract_mean:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            inputs -= tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
            inputs += self.bias

        variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon)
