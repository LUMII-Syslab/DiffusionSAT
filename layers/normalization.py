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

    def call(self, inputs, graph: tf.SparseTensor = None, **kwargs):
        """
        :param graph: graph level adjacency matrix
        :param count_in_graph: element count in each graph
        :param inputs: input tensor variables or clauses state
        """
        mask = graph.indices[:, 0] if graph is not None else None

        # input size: cells x feature_maps
        if self.subtract_mean and graph is not None:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            mean = tf.sparse.sparse_dense_matmul(graph, inputs)
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

class VariablesNeighborNorm(tf.keras.layers.Layer):
    """ Normalize variables by subtracting neighbor mean and then normalize by features axis
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        super(VariablesNeighborNorm, self).__init__(**kwargs)


    def call(self, variables, adj_matrix: tf.SparseTensor = None, **kwargs):
        """
        :param inputs: input tensor variables
        """
        # subtract neighbor mean
        literals = tf.concat([variables, variables], axis=0)
        literals1 = tf.concat([literals, tf.ones([tf.shape(literals)[0], 1])], axis=1) #todo: it is possible to precompute degree
        clauses_val = tf.sparse.sparse_dense_matmul(adj_matrix, literals1)
        lit_new = tf.sparse.sparse_dense_matmul(adj_matrix, clauses_val, adjoint_a=True)
        lit1, lit2 = tf.split(lit_new, 2, axis=0)
        var_new_deg = lit1 + lit2
        var_new = var_new_deg[:, :-1]
        deg = var_new_deg[:, -1:]
        mean = var_new / tf.maximum(deg, 2)  # 2 is to avoid degenerate case with a single unit clause
        variables -= mean                    # todo: better treatment of self references

        variance = tf.reduce_mean(tf.square(variables), axis=1, keepdims=True)

        return variables * tf.math.rsqrt(variance + self.epsilon)

class ClausesNeighborNorm(tf.keras.layers.Layer):
    """ Normalize clauses by subtracting neighbor mean and then normalize by features axis
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        self.epsilon = epsilon
        super(ClausesNeighborNorm, self).__init__(**kwargs)


    def call(self, clauses, cl_adj_matrix: tf.SparseTensor = None, **kwargs):
        """
        :param inputs: input tensor variables
        """
        # subtract neighbor mean
        clauses1 = tf.concat([clauses, tf.ones([tf.shape(clauses)[0], 1])], axis=1) #todo: it is possible to precompute degree
        lit_val = tf.sparse.sparse_dense_matmul(cl_adj_matrix, clauses1)
        clauses_new_deg = tf.sparse.sparse_dense_matmul(cl_adj_matrix, lit_val, adjoint_a=True)
        clause_new = clauses_new_deg[:, :-1]
        deg = clauses_new_deg[:, -1:]
        mean = clause_new / tf.maximum(deg, 2) # todo: better treatment of self references
        clauses -= mean

        variance = tf.reduce_mean(tf.square(clauses), axis=1, keepdims=True)
        return clauses * tf.math.rsqrt(variance + self.epsilon)
