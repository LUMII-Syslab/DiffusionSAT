import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense


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

        self.query_layer = Dense(hidden_nmaps, activation=tfa.activations.gelu)
        self.key_layer = Dense(hidden_nmaps, activation=tfa.activations.gelu)
        self.value_layer = Dense(output_nmaps, activation=tfa.activations.gelu)

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

        coef = tf.matmul(q, tf.linalg.matrix_transpose(k))  # result [n, m]
        coef = coef / tf.sqrt(tf.cast(self.hidden_nmaps, tf.float32))  # result [n, m]
        coef = coef * adj_matrix

        out = tf.sparse.softmax(coef)  # result [n, m]
        # out = tf.layers.dropout(out, training=self._is_training)
        return tf.sparse.sparse_dense_matmul(out, v)  # result [n, output_nmaps]
