import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_addons as tfa


def matmul_with_sparse_mask(a: tf.Tensor, b: tf.Tensor, mask: tf.SparseTensor, scale=1, activation=None):
    """ Should give the same result as matmul(a,transpose(b))*mask, but calculation is significantly faster.
    """
    a_val = tf.gather(a, mask.indices[:, 0])
    b_val = tf.gather(b, mask.indices[:, 1])
    dot = tf.reduce_sum(a_val * b_val, axis=-1)

    if activation:
        dot = activation(dot * scale)

    return tf.SparseTensor(mask.indices, dot, dense_shape=mask.dense_shape)


class GraphAttentionLayer(tf.keras.layers.Layer):
    """
    Simple scaled dot-product attention for graph neural network.
    Attention coefficients are only calculated for neighbor nodes,
    rest of the nodes are masked out using adjacency matrix.
    """

    def __init__(self, hidden_nmaps, output_nmaps, activation=tf.nn.relu, **kwargs):
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
        :param kwargs:
        :return:
        """

        q = self.query_layer(query)
        k = self.key_layer(memory)
        v = self.value_layer(memory)

        q = tf.split(q, num_or_size_splits=self.heads, axis=-1)
        k = tf.split(k, num_or_size_splits=self.heads, axis=-1)
        v = tf.split(v, num_or_size_splits=self.heads, axis=-1)

        results = []
        for i in range(self.heads):
            if self.use_sparse_mul:
                scale = 1 / tf.sqrt(tf.cast(self.hidden_nmaps, tf.float32))
                coef = matmul_with_sparse_mask(q[i], k[i], adj_matrix, scale)
                coef = tf.sparse.softmax(tf.sparse.transpose(coef))  # result [n, m]
            else:
                coef = tf.matmul(q, tf.transpose(k))  # result [n, m]
                coef = coef / tf.sqrt(tf.cast(self.hidden_nmaps, tf.float32))  # result [n, m]
                coef = coef * adj_matrix
                coef = tf.sparse.softmax(coef)  # result [n, m]

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
