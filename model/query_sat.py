import tensorflow as tf
from tensorflow.keras.models import Model

from loss.sat import variables_mul_loss
from model.mlp import MLP


class QuerySAT(Model):

    def __init__(self, feature_maps=512, msg_layers=3, vote_layers=3, rounds=16, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.rounds = rounds

        self.literals_norm = tf.keras.layers.LayerNormalization()

        self.literals_update = MLP(msg_layers, feature_maps, feature_maps,
                                   out_activation=tf.nn.relu,
                                   name="literals_update")

        self.forget_gate = MLP(msg_layers, feature_maps, feature_maps,
                               out_activation=tf.sigmoid,
                               name="clauses_update")

        self.literals_vote = MLP(vote_layers, feature_maps, 1, name="literals_vote")
        self.literals_query = MLP(vote_layers, feature_maps, feature_maps, name="literals_query")
        self.literals_query_inter = MLP(vote_layers, feature_maps, feature_maps, name="literals_query_inter")

        self.feature_maps = feature_maps

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=bool)])
    def call(self, adj_matrix, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse adjacency matrix
        n_lits = shape[0]
        n_clauses = shape[1]
        n_vars = n_lits // 2

        literals = tf.random.truncated_normal([n_lits, self.feature_maps], stddev=0.25)

        for _ in tf.range(self.rounds):
            variables = tf.concat([literals[:n_vars], literals[n_vars:]], axis=1)  # n_vars x 2
            logits = self.literals_query(variables)
            clauses_loss = variables_mul_loss(logits, clauses)
            clauses_loss = self.literals_query_inter(clauses_loss)

            literals_loss = tf.sparse.sparse_dense_matmul(adj_matrix, clauses_loss)

            unit = tf.concat([literals, literals_loss], axis=-1)
            unit = self.flip(unit, n_vars)
            unit = self.literals_norm(unit)

            forget_gate = self.forget_gate(unit)
            literals_new = self.literals_update(unit)

            literals = (1 - forget_gate) * literals + forget_gate * literals_new

        variables = tf.concat([literals[:n_vars], literals[n_vars:]], axis=1)  # n_vars x 2
        logits = self.literals_vote(variables)
        return tf.squeeze(logits, axis=[-1])  # Size of n_vars

    @staticmethod
    def flip(literals, n_vars):
        return tf.concat([literals[n_vars:(2 * n_vars), :], literals[0:n_vars, :]], axis=0)
