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
        pass
        # num_units = input_shape.as_list()[-1]
        # if self.subtract_mean:
        #     self.bias = self.add_weight("bias", [num_units], initializer=tf.zeros_initializer)

    def call(self, inputs, graph: tf.SparseTensor = None, count_in_graph: tf.Tensor = None, **kwargs):
        """
        :param graph: graph level adjacency matrix
        :param count_in_graph: element count in each graph
        :param inputs: input tensor variables or clauses state
        """
        if count_in_graph is not None:
            count_in_graph = tf.cast(count_in_graph, tf.float32)
            count_in_graph = tf.expand_dims(count_in_graph, axis=-1)

        mask = graph.indices[:, 0] if graph is not None else None

        # input size: cells x feature_maps
        if self.subtract_mean and graph is not None:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            mean = tf.sparse.sparse_dense_matmul(graph, inputs) / count_in_graph
            # mean = tf.math.unsorted_segment_mean(inputs, graph_mask, tf.reduce_max(graph_mask) + 1)
            # mean = tf.math.segment_mean(inputs, graph_mask)

            inputs -= tf.gather(mean, mask)
            # inputs += self.bias

        # input size: cells x feature_maps
        # nb here we deviate from PairNorm, we use axis=0, PairNorm uses axis=1
        # if graph is not None:
        #     variance = tf.sparse.sparse_dense_matmul(graph, tf.square(inputs)) / count_in_graph
        #     # variance = tf.math.unsorted_segment_mean(tf.square(inputs), graph_mask, tf.reduce_max(graph_mask) + 1)
        #     variance = tf.gather(variance, mask)
        # else:
        variance = tf.reduce_mean(tf.square(inputs), axis=1, keepdims=True)

        return inputs * tf.math.rsqrt(variance + self.epsilon) # +self.bias
