import math

import tensorflow as tf
from tensorflow.keras.models import Model

from loss.sat import softplus_mixed_loss
from model.mlp import MLP
from utils.parameters_log import *


class SimpleNeuroSAT(Model):
    def __init__(self, optimizer, **kwargs):
        super(SimpleNeuroSAT, self).__init__(**kwargs)
        self.optimizer = optimizer

        self.feature_maps = 80
        self.n_rounds = 32
        self.norm_axis = 0
        self.norm_eps = 1e-3
        self.n_update_layers = 1
        self.n_score_layers = 3

        self.L_updates = [MLP(self.n_update_layers + 1, 2 * self.feature_maps + self.feature_maps, self.feature_maps,
                              activation=tf.nn.relu6, name="L_u")] * self.n_rounds
        self.C_updates = [MLP(self.n_update_layers + 1, self.feature_maps + self.feature_maps, self.feature_maps,
                              activation=tf.nn.relu6, name="C_u")] * self.n_rounds

        init = tf.constant_initializer(1.0 / math.sqrt(self.feature_maps))
        self.L_init_scale = self.add_weight(name="L_init_scale", shape=[], initializer=init)
        self.C_init_scale = self.add_weight(name="C_init_scale", shape=[], initializer=init)

        self.LC_scale = self.add_weight(name="LC_scale", shape=[], initializer=tf.constant_initializer(0.1))
        self.CL_scale = self.add_weight(name="CL_scale", shape=[], initializer=tf.constant_initializer(0.1))

        self.V_score = MLP(self.n_score_layers + 1, 2 * self.feature_maps, 1, activation=tf.nn.relu6, name="V_score")

    def call(self, adj_matrix, clauses, clauses_count, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse factor matrix
        n_lits = shape[0]
        n_vars = n_lits // 2
        n_clauses = shape[1]

        L = tf.ones(shape=[2 * n_vars, self.feature_maps], dtype=tf.float32) * self.L_init_scale
        C = tf.ones(shape=[n_clauses, self.feature_maps], dtype=tf.float32) * self.C_init_scale
        CL = tf.sparse.transpose(adj_matrix)

        graph_count = tf.shape(clauses_count)
        graph_id = tf.range(0, graph_count[0])
        clauses_mask = tf.repeat(graph_id, clauses_count)

        loss = 0

        def flip(lits):
            return tf.concat([lits[n_vars:, :], lits[0:n_vars, :]], axis=0)

        for t in range(self.n_rounds):
            LC_msgs = tf.sparse.sparse_dense_matmul(CL, L) * self.LC_scale
            C = self.C_updates[t](tf.concat([C, LC_msgs], axis=-1))
            C = tf.debugging.check_numerics(C, message="C after update")
            C = normalize(C, axis=self.norm_axis, eps=self.norm_eps)
            C = tf.debugging.check_numerics(C, message="C after norm")

            CL_msgs = tf.sparse.sparse_dense_matmul(adj_matrix, C) * self.CL_scale
            L = self.L_updates[t](tf.concat([L, CL_msgs, flip(L)], axis=-1))
            L = tf.debugging.check_numerics(L, message="L after update")
            L = normalize(L, axis=self.norm_axis, eps=self.norm_eps)
            L = tf.debugging.check_numerics(L, message="L after norm")

            V = tf.concat([L[0:n_vars, :], L[n_vars:, :]], axis=1)
            logits = self.V_score(V)  # (n_vars, 1)

            per_clause_loss = softplus_mixed_loss(logits, clauses)
            per_graph_loss = tf.math.segment_sum(per_clause_loss, clauses_mask)
            loss += tf.reduce_sum(tf.sqrt(per_graph_loss + 1e-6))

        return logits, loss / self.n_rounds

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def train_step(self, adj_matrix, clauses, variable_count, clauses_count):
        with tf.GradientTape() as tape:
            logits, loss = self.call(adj_matrix, clauses, clauses_count, training=True)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": loss,
            "gradients": gradients
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def predict_step(self, adj_matrix, clauses, variable_count, clauses_count):
        predictions, loss = self.call(adj_matrix, clauses, clauses_count, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_TRAIN_ROUNDS: self.n_rounds,
                HP_TEST_ROUNDS: self.n_rounds,
                HP_MLP_LAYERS: self.n_update_layers
                }


def normalize(x, axis, eps):
    mean, variance = tf.nn.moments(x, axes=[axis], keepdims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=eps)
