import math

import tensorflow as tf
from tensorflow.keras.models import Model

from loss.sat import softplus_mixed_loss
from model.mlp import MLP
from utils.parameters_log import *


class SimpleNeuroSAT(Model):
    def __init__(self):
        LC_scale = 0.1
        CL_scale = 0.1
        self.feature_maps = 80

        self.n_rounds = 4
        self.norm_axis = 0
        self.norm_eps = 1e-3
        self.n_update_layers = 1
        self.n_score_layers = 3

        self.L_updates = [MLP(self.n_update_layers + 1, 2 * self.feature_maps + self.feature_maps, self.feature_maps,
                              activation=tf.nn.relu6, name="L_u")] * self.n_rounds
        self.C_updates = [MLP(self.n_update_layers + 1, self.feature_maps + self.feature_maps, self.feature_maps,
                              activation=tf.nn.relu6, name="C_u")] * self.n_rounds

        init = tf.constant_initializer(1.0 / math.sqrt(self.feature_maps))
        self.L_init_scale = self.get_variable(name="L_init_scale", shape=[], initializer=init)
        self.C_init_scale = self.get_variable(name="C_init_scale", shape=[], initializer=init)

        self.LC_scale = self.get_variable(name="LC_scale", shape=[], initializer=tf.constant_initializer(LC_scale))
        self.CL_scale = self.get_variable(name="CL_scale", shape=[], initializer=tf.constant_initializer(CL_scale))

        self.V_score = MLP(self.n_score_layers + 1, 2 * self.feature_maps, 1, activation=tf.nn.relu6, name="V_score")

    def call(self, adj_matrix, clauses, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse factor matrix
        n_lits = shape[0]
        n_vars = n_lits // 2
        n_clauses = shape[1]

        L = tf.ones(shape=[2 * n_vars, self.feature_maps], dtype=tf.float32) * self.L_init_scale
        C = tf.ones(shape=[n_clauses, self.feature_maps], dtype=tf.float32) * self.C_init_scale

        LC = tf.sparse.transpose(adj_matrix)

        def flip(lits):
            return tf.concat([lits[n_vars:, :], lits[0:n_vars, :]], axis=0)

        for t in range(self.n_rounds):
            C_old, L_old = C, L

            LC_msgs = tf.sparse.sparse_dense_matmul(adj_matrix, L) * self.LC_scale
            C = self.C_updates[t].forward(tf.concat([C, LC_msgs], axis=-1))
            C = tf.debugging.check_numerics(C, message="C after update")
            C = normalize(C, axis=self.norm_axis, eps=self.norm_eps)
            C = tf.debugging.check_numerics(C, message="C after norm")

            CL_msgs = tf.sparse.sparse_dense_matmul(LC, C) * self.CL_scale
            L = self.L_updates[t].forward(tf.concat([L, CL_msgs, flip(L)], axis=-1))
            L = tf.debugging.check_numerics(L, message="L after update")
            L = normalize(L, axis=self.norm_axis, eps=self.norm_eps)
            L = tf.debugging.check_numerics(L, message="L after norm")

        V = tf.concat([L[0:n_vars, :], L[n_vars:, :]], axis=1)
        V_scores = self.V_score.forward(V)  # (n_vars, 1)

        logits_loss = tf.reduce_sum(softplus_mixed_loss(V_scores, clauses))

        return V_scores, logits_loss

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def train_step(self, adj_matrix, clauses, variable_count, clauses_count):
        with tf.GradientTape() as tape:
            logits, loss = self.call(adj_matrix, clauses, training=True)
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
        predictions, loss = self.call(adj_matrix, clauses, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_TRAIN_ROUNDS: self.rounds,
                HP_TEST_ROUNDS: self.rounds,
                HP_MLP_LAYERS: self.vote_layers
                }


def normalize(x, axis, eps):
    mean, variance = tf.nn.moments(x, axes=[axis], keep_dims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=eps)
