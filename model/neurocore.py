import math

import tensorflow as tf
from tensorflow.keras.models import Model

from loss.sat import softplus_mixed_loss_adj
from model.mlp import MLP
from utils.parameters_log import *
from utils.sat import is_batch_sat


class NeuroCore(Model):
    def __init__(self, optimizer, test_rounds=64, train_rounds=32, **kwargs):
        super(NeuroCore, self).__init__(**kwargs)
        self.optimizer = optimizer

        self.feature_maps = 128
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.norm_axis = 0
        self.norm_eps = 1e-6
        self.n_update_layers = 2
        self.n_score_layers = 2

        self.L_updates = MLP(self.n_update_layers + 1, 2 * self.feature_maps + self.feature_maps, self.feature_maps,
                             activation=tf.nn.relu6, name="L_u")
        self.C_updates = MLP(self.n_update_layers + 1, self.feature_maps + self.feature_maps, self.feature_maps,
                             activation=tf.nn.relu6, name="C_u")

        init = tf.constant_initializer(1.0 / math.sqrt(self.feature_maps))
        self.L_init_scale = self.add_weight(name="L_init_scale", shape=[], initializer=init)
        self.C_init_scale = self.add_weight(name="C_init_scale", shape=[], initializer=init)

        self.LC_scale = self.add_weight(name="LC_scale", shape=[], initializer=tf.constant_initializer(0.1))
        self.CL_scale = self.add_weight(name="CL_scale", shape=[], initializer=tf.constant_initializer(0.1))

        self.variables_query = MLP(self.n_update_layers + 1, self.feature_maps, self.feature_maps,
                                   activation=tf.nn.relu6, name="variables_query", do_layer_norm=False)
        self.V_score = MLP(self.n_score_layers + 1, 2 * self.feature_maps, 1, activation=tf.nn.relu6, name="V_score")

        self.self_supervised = False

    def call(self, adj_matrix, clauses_graph, variables_graph, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse factor matrix
        n_lits = shape[0]
        n_vars = n_lits // 2
        n_clauses = shape[1]

        L = tf.ones(shape=[n_lits, self.feature_maps], dtype=tf.float32) * self.L_init_scale
        C = tf.ones(shape=[n_clauses, self.feature_maps], dtype=tf.float32) * self.C_init_scale
        loss = 0.
        supervised_loss = 0.

        def flip(lits):
            return tf.concat([lits[n_vars:, :], lits[0:n_vars, :]], axis=0)

        rounds = self.train_rounds if training else self.test_rounds
        logits = tf.zeros([n_vars, 1])
        cl_adj_matrix = tf.sparse.transpose(adj_matrix)

        for steps in tf.range(rounds):
            LC_msgs = tf.sparse.sparse_dense_matmul(cl_adj_matrix, L) * self.LC_scale
            C = self.C_updates(tf.concat([C, LC_msgs], axis=-1))
            C = tf.debugging.check_numerics(C, message="C after update")
            C = normalize(C, axis=self.norm_axis, eps=self.norm_eps)
            C = tf.debugging.check_numerics(C, message="C after norm")

            CL_msgs = tf.sparse.sparse_dense_matmul(adj_matrix, C) * self.CL_scale
            L = self.L_updates(tf.concat([L, CL_msgs, flip(L)], axis=-1))
            L = tf.debugging.check_numerics(L, message="L after update")
            L = normalize(L, axis=self.norm_axis, eps=self.norm_eps)
            L = tf.debugging.check_numerics(L, message="L after norm")

            v, v_n = tf.split(L, 2, axis=0)
            logits = self.V_score(tf.concat([v, v_n], axis=-1))  # (n_vars, 1)
            is_sat = is_batch_sat(logits, cl_adj_matrix)
            if is_sat == 1:
                break

            per_clause_loss = softplus_mixed_loss_adj(logits, cl_adj_matrix)
            per_graph_loss = tf.sparse.sparse_dense_matmul(clauses_graph, per_clause_loss)
            loss += tf.reduce_sum(tf.sqrt(per_graph_loss + 1e-6))

        return logits, loss / tf.cast(rounds, tf.float32) + supervised_loss, steps

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def train_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        with tf.GradientTape() as tape:
            logits, loss, steps = self.call(adj_matrix, clauses_graph, variables_graph, training=True)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": loss,
            "gradients": gradients,
            "steps_taken": steps
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix, clauses_graph, variables_graph, solutions):
        predictions, loss, steps = self.call(adj_matrix, clauses_graph, variables_graph, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1),
            "steps_taken": steps
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_TRAIN_ROUNDS: self.train_rounds,
                HP_TEST_ROUNDS: self.test_rounds,
                HP_MLP_LAYERS: self.n_update_layers
                }


def normalize(x, axis, eps):
    mean, variance = tf.nn.moments(x, axes=[axis], keepdims=True)
    return tf.nn.batch_normalization(x, mean, variance, offset=None, scale=None, variance_epsilon=eps)
