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


class PairNorm(tf.keras.layers.Layer):
    """ PairNorm: Tackling Oversmoothing in GNNs https://arxiv.org/abs/1909.12223
    """

    def __init__(self, epsilon=1e-6, subtract_mean=False, **kwargs):
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean
        super(PairNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        num_units = input_shape.as_list()[-1]
        if self.subtract_mean:
            self.bias = self.add_weight("bias", [num_units], initializer=tf.zeros_initializer)

    def call(self, inputs, graph_mask=None, **kwargs):
        """
        :param inputs: input tensor variables or clauses state
        :param graph_mask: 1-D mask for separate graphs. Take note that for clauses and variables this values differs.
        """
        # input size: cells x feature_maps
        if self.subtract_mean and graph_mask is not None:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            #mean = tf.math.unsorted_segment_mean(inputs, graph_mask, tf.reduce_max(graph_mask) + 1)
            mean = tf.math.segment_mean(inputs, graph_mask)
            inputs -= tf.gather(mean, graph_mask)
            inputs += self.bias

        # input size: cells x feature_maps
        # nb here we deviate from PairNorm, we use axis=0, PairNorm uses axis=1
        if graph_mask is not None:
            variance = tf.math.segment_mean(tf.square(inputs), graph_mask)
            variance = tf.gather(variance, graph_mask)
        else:
            variance = tf.reduce_mean(tf.square(inputs), axis=0, keepdims=True)

        return inputs * tf.math.rsqrt(variance + self.epsilon)
