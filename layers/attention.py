import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense

from layers.layer_normalization import LayerNormalization


def matmul_with_sparse_mask(a: tf.Tensor, b: tf.Tensor, mask: tf.SparseTensor, scale=1):
    """ Should give the same result as matmul(a,transpose(b))*mask, but calculation is significantly faster.
    TODO: Check that there is no error!
    """
    a_val = tf.gather(a, mask.indices[:, 0])
    b_val = tf.gather(b, mask.indices[:, 1])
    dot = tf.reduce_sum(a_val * b_val, axis=-1)
    dot = tf.sigmoid(dot * scale)

    return tf.SparseTensor(mask.indices, dot, dense_shape=mask.dense_shape)


class GraphAttentionLayer(tf.keras.layers.Layer):
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

        self.query_layer = Dense(hidden_nmaps, activation=tfa.activations.gelu)
        self.key_layer = Dense(hidden_nmaps, activation=tfa.activations.gelu)
        self.value_layer = Dense(output_nmaps, activation=tfa.activations.gelu)
        self.layer_norm = LayerNormalization()

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

        if self.use_sparse_mul:
            scale = 1 / tf.sqrt(tf.cast(self.hidden_nmaps, tf.float32))
            coef = matmul_with_sparse_mask(q, k, adj_matrix, scale)
        else:
            coef = tf.matmul(q, tf.transpose(k))  # result [n, m]
            coef = coef / tf.sqrt(tf.cast(self.hidden_nmaps, tf.float32))  # result [n, m]
            coef = coef * adj_matrix

        # out = tf.sparse.softmax(coef)  # result [n, m]
        result = tf.sparse.sparse_dense_matmul(coef, v)  # result [n, output_nmaps]
        return result