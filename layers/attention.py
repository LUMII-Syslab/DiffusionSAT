import tensorflow as tf
from tensorflow.keras.layers import Dense

from model.mlp import MLP


def matmul_with_sparse_mask(a: tf.Tensor, b: tf.Tensor, mask: tf.SparseTensor, scale=1):
    """ Gives the same result as matmul(a,transpose(b))*mask, but calculation
     is significantly faster for sparse masks.
    """
    a_val = tf.gather(a, mask.indices[:, 0])
    b_val = tf.gather(b, mask.indices[:, 1])
    dot = tf.reduce_sum(a_val * b_val, axis=-1)
    dot = dot * scale

    return tf.SparseTensor(mask.indices, dot, dense_shape=mask.dense_shape)


class DotAttentionLayer(tf.keras.layers.Layer):
    """
    Simple scaled dot-product attention for graph neural network.
    Attention coefficients are only calculated for neighbor nodes,
    rest of the nodes are masked out using adjacency matrix.
    """

    def __init__(self, hidden_nmaps, output_nmaps, activation=tf.nn.leaky_relu, **kwargs):
        super().__init__(**kwargs)
        self.hidden_nmaps = hidden_nmaps
        self.output_nmaps = output_nmaps
        self.activation = activation
        self.use_sparse_mul = True
        self.heads = 4

        self.query_layer = Dense(hidden_nmaps, activation=activation)
        self.key_layer = Dense(hidden_nmaps, activation=activation)
        self.value_layer = Dense(output_nmaps, activation=activation)

        self.output_weight = Dense(output_nmaps)

    def call(self, query: tf.Tensor, memory: tf.Tensor, adj_matrix: tf.sparse.SparseTensor, **kwargs):
        """
        :param query: [n, nmaps_1]
        :param memory: [m, nmaps_2]
        :param adj_matrix: sparse matrix with dense shape [n, m]
        :return: softmax(matmul(q,transpose(k))*adj_matrix)*v : [n, output_nmaps]
        """

        q = self.query_layer(query)
        k = self.key_layer(memory)
        v = self.value_layer(memory)

        q = tf.split(q, num_or_size_splits=self.heads, axis=-1)
        k = tf.split(k, num_or_size_splits=self.heads, axis=-1)
        v = tf.split(v, num_or_size_splits=self.heads, axis=-1)

        results = []
        for i in range(self.heads):  # TODO: Can this be done in parallel?
            if self.use_sparse_mul:
                scale = 1 / tf.sqrt(tf.cast(self.hidden_nmaps // self.heads, tf.float32))
                coef = matmul_with_sparse_mask(q[i], k[i], adj_matrix, scale)
                coef = tf.sparse.softmax(tf.sparse.transpose(coef))  # result [n, m]
            else:
                coef = tf.matmul(q[i], tf.transpose(k[i]))  # result [n, m]
                coef = coef / tf.sqrt(tf.cast(self.hidden_nmaps // self.heads, tf.float32))  # result [n, m]
                coef = coef * adj_matrix
                coef = tf.sparse.softmax(tf.sparse.transpose(coef))  # result [n, m]

            res = tf.sparse.sparse_dense_matmul(coef, v[i], adjoint_a=True)  # result [n, output_nmaps]
            results.append(res)

            # coef = tf.sparse.reorder(coef)
            # image = tf.sparse.slice(coef, [0, 0], [128, 256])
            # image = tf.sparse.to_dense(image)
            # image = image / tf.math.reduce_max(image)
            # image = tf.expand_dims(image, axis=-1)
            # image = tf.expand_dims(image, axis=0)
            # tf.summary.image(f"coef_head_{i}", image)

        output = tf.concat(results, axis=-1)
        return self.output_weight(output)


class AdditiveAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_maps, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.heads = 1
        self.pre_process = MLP(3, hidden_maps, self.heads, do_layer_norm=True)
        self.memory_mlp = MLP(3, hidden_maps, hidden_maps, do_layer_norm=True)

    def call(self, query: tf.Tensor, memory: tf.Tensor, adj_matrix: tf.sparse.SparseTensor, **kwargs):
        q = tf.gather(query, adj_matrix.indices[:, 0])
        k = tf.gather(memory, adj_matrix.indices[:, 1])
        units = tf.concat([q, k], axis=-1)
        units = self.pre_process(units)
        units = tf.tanh(units)

        unit_heads = tf.split(units, self.heads, axis=-1)
        mem_head = self.memory_mlp(memory)
        mem_head = tf.split(mem_head, self.heads, axis=-1)

        results = []
        for unit, mem in zip(unit_heads, mem_head):
            h = tf.squeeze(unit, axis=-1)
            weighted_adj = tf.SparseTensor(adj_matrix.indices, h, dense_shape=adj_matrix.dense_shape)
            tf.summary.histogram("attention_weights", weighted_adj.values)
            result = tf.sparse.sparse_dense_matmul(weighted_adj, mem)
            results.append(result)

        # rand = tf.random.normal(())
        # if rand > 0.8:
        #     image = tf.sparse.slice(tf.sparse.transpose(weighted_adj), [0, 0], [512, 512])
        #     image = tf.sparse.to_dense(image)
        #     image = tf.expand_dims(image, axis=-1)
        #     image = tf.expand_dims(image, axis=0)
        #     tf.summary.image("after_softmax", image / tf.reduce_max(image))

        return tf.concat(results, axis=-1)

class SigmoidAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_maps, **kwargs):
        super(SigmoidAttention, self).__init__(**kwargs)
        self.unit_mlp = MLP(3, hidden_maps, hidden_maps, do_layer_norm = True)

    def call(self, query: tf.Tensor, memory: tf.Tensor, adj_matrix: tf.sparse.SparseTensor, **kwargs):
        q = tf.gather(query, adj_matrix.indices[:, 0])
        k = tf.gather(memory, adj_matrix.indices[:, 1])
        units = tf.concat([q, k], axis=-1)
        weights = tf.sigmoid(self.unit_mlp(units))
        #lit_val = tf.math.segment_sum(units, adj_matrix.indices[:,1]) # are indices sorted?
        lit_val = tf.math.unsorted_segment_sum(k * weights, adj_matrix.indices[:, 0], adj_matrix.dense_shape[0])
        return lit_val
