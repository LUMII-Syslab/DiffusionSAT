import tensorflow as tf


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, axis=1, epsilon=1e-6, subtract_mean=False, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.subtract_mean:
            num_units = input_shape.as_list()[-1]
            self.bias = self.add_weight("bias", [num_units], initializer=tf.zeros_initializer)

    def call(self, inputs, **kwargs):
        if self.subtract_mean:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            inputs -= tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
            inputs += self.bias

        variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon)


class PairNorm(tf.keras.layers.Layer):
    """
    PairNorm: Tackling Oversmoothing in GNNs https://arxiv.org/abs/1909.12223
    """
    def __init__(self, epsilon=1e-6, subtract_mean=False, **kwargs):
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean
        super(PairNorm, self).__init__(**kwargs)

    def call(self, inputs, graph: tf.SparseTensor = None, **kwargs):
        """
        :param graph: graph level adjacency matrix
        :param count_in_graph: element count in each graph
        :param inputs: input tensor variables or clauses state
        """
        mask = graph.indices[:, 0] if graph is not None else None

        # input size: cells x feature_maps
        if self.subtract_mean:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            if graph is not None:
                mean = tf.sparse.sparse_dense_matmul(graph, inputs)
                inputs -= tf.gather(mean, mask)
            else:  # assume one graph per batch
                mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
                inputs -= mean

        variance = tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True)

        return inputs * tf.math.rsqrt(variance + self.epsilon)
