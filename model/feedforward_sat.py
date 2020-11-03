import tensorflow as tf
from tensorflow.keras.models import Model

from model.mlp import MLP


class FeedForwardSAT(Model):

    def __init__(self, feature_maps=256, msg_layers=3, vote_layers=3, rounds=16, **kwargs):
        super().__init__(**kwargs, name="ResidualSAT")
        self.rounds = rounds

        init = tf.initializers.RandomNormal()
        self.L_init = self.add_weight(name="L_init", shape=[1, feature_maps], initializer=init, trainable=False)
        self.C_init = self.add_weight(name="C_init", shape=[1, feature_maps], initializer=init, trainable=False)

        self.L_norm = tf.keras.layers.LayerNormalization()
        self.C_norm = tf.keras.layers.LayerNormalization()

        self.liters_update = MLP(msg_layers, feature_maps, feature_maps, name="Lit_update")
        self.clauses_update = MLP(msg_layers, feature_maps, feature_maps, name="Clauses_update")

        self.L_vote = MLP(vote_layers, feature_maps, 1, name="L_vote")

        self.denom = tf.sqrt(tf.cast(feature_maps, tf.float32))
        self.feature_maps = feature_maps

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self, inputs, labels=None, training=None, mask=None):
        shape = tf.shape(inputs)  # inputs is sparse adjacency matrix
        n_lits = shape[0]
        n_clauses = shape[1]
        n_vars = n_lits // 2

        literals = tf.tile(self.L_init / self.denom, [n_lits, 1])
        clauses = tf.tile(self.C_init / self.denom, [n_clauses, 1])

        for _ in tf.range(self.rounds):
            clauses_new = tf.sparse.sparse_dense_matmul(inputs, literals, adjoint_a=True)
            clauses = self.clauses_update(tf.concat([clauses_new, clauses], axis=-1))
            clauses = self.C_norm(clauses)

            literals_new = tf.sparse.sparse_dense_matmul(inputs, clauses)
            literals = self.liters_update(tf.concat([literals_new, literals], axis=-1))
            literals = self.L_norm(literals)

        variables = tf.concat([literals[:n_vars], literals[n_vars:]], axis=1)  # n_vars x 2
        logits = self.L_vote(variables)
        return tf.squeeze(logits, axis=[-1])  # Size of n_vars

    @staticmethod
    def flip(literals, n_vars):
        return tf.concat([literals[n_vars:(2 * n_vars), :], literals[0:n_vars, :]], axis=0)
