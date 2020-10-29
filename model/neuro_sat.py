import tensorflow as tf
from tensorflow.keras.layers import LSTMCell
from tensorflow.keras.models import Model

from model.mlp import MLP


class NeuroSAT(Model):

    def __init__(self, feature_maps=128, msg_layers=3, vote_layers=3, rounds=16, **kwargs):
        super().__init__(**kwargs, name="NeuroSAT")
        self.rounds = rounds

        init = tf.initializers.RandomNormal()
        self.L_init = self.add_weight(name="L_init", shape=[1, feature_maps], initializer=init, trainable=True)
        self.C_init = self.add_weight(name="C_init", shape=[1, feature_maps], initializer=init, trainable=True)

        self.LC_msg = MLP(msg_layers, feature_maps, feature_maps, name="LC_msg")
        self.CL_msg = MLP(msg_layers, feature_maps, feature_maps, name="CL_msg")

        self.L_update = LSTMCell(feature_maps, name="L_update")
        self.C_update = LSTMCell(feature_maps, name="C_update")

        self.L_vote = MLP(vote_layers, feature_maps * 2, 1, name="L_vote")

        self.denom = tf.sqrt(tf.cast(feature_maps, tf.float32))
        self.feature_maps = feature_maps

    def call(self, inputs, labels=None, training=None, mask=None):
        shape = tf.shape(inputs)  # inputs is sparse adjacency matrix
        n_lits = shape[0]
        n_clauses = shape[1]
        n_vars = n_lits // 2

        l_output = tf.tile(self.L_init / self.denom, [n_lits, 1])
        c_output = tf.tile(self.C_init / self.denom, [n_clauses, 1])

        l_state = [l_output, tf.zeros([n_lits, self.feature_maps])]
        c_state = [c_output, tf.zeros([n_clauses, self.feature_maps])]

        for _ in tf.range(self.rounds):
            LC_pre_msgs = self.LC_msg(l_state[0])
            LC_msgs = tf.sparse.sparse_dense_matmul(inputs, LC_pre_msgs, adjoint_a=True)

            _, c_state = self.C_update(inputs=LC_msgs, states=c_state)

            CL_pre_msgs = self.CL_msg(c_state[0])
            CL_msgs = tf.sparse.sparse_dense_matmul(inputs, CL_pre_msgs)

            _, l_state = self.L_update(inputs=tf.concat([CL_msgs, self.flip(l_state[0], n_vars)], axis=1),
                                       states=l_state)

        literals = l_state[0]

        variables = tf.concat([literals[:n_vars], literals[n_vars:]], axis=1)  # n_vars x 2
        logits = self.L_vote(variables)
        return tf.squeeze(logits, axis=[-1])  # Size of n_vars

    @staticmethod
    def flip(literals, n_vars):
        return tf.concat([literals[n_vars:(2 * n_vars), :], literals[0:n_vars, :]], axis=0)
