import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.normalization import PairNorm
from loss.sat import softplus_loss, unsat_clause_count, softplus_log_loss, softplus_mixed_loss
from model.mlp import MLP
from utils.parameters_log import *


class QuerySATLit(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=128, msg_layers=3,
                 vote_layers=3, train_rounds=32, test_rounds=64,
                 query_maps=32, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.optimizer = optimizer
        self.vote_layers = vote_layers

        self.add_gradient = True
        self.use_message_passing = False

        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)

        self.clauses_update = MLP(vote_layers, feature_maps * 3, feature_maps + 1 * query_maps, name="clause_update", do_layer_norm=False)
        self.literals_update = MLP(vote_layers, feature_maps * 2, feature_maps, name="variables_update", do_layer_norm=False)

        self.literals_output = MLP(vote_layers, feature_maps, 1, name="variables_output", do_layer_norm=False)
        self.literals_query = MLP(msg_layers, query_maps * 2, query_maps * 2, name="variables_query", do_layer_norm=False)

        self.feature_maps = feature_maps
        self.query_maps = query_maps

    def zero_state(self, n_units, n_features, stddev=0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        onehot = onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev
        return onehot

    def call(self, adj_matrix, clauses=None, variable_count=None, clauses_count=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)
        n_literals = shape[0]
        n_vars = n_literals // 2
        n_clauses = shape[1]

        graph_count = tf.shape(variable_count)
        graph_id = tf.range(0, graph_count[0])
        variables_mask = tf.repeat(graph_id, variable_count)
        literals_mask = tf.concat([variables_mask, variables_mask], axis=0)
        clauses_mask = tf.repeat(graph_id, clauses_count)

        literals = self.zero_state(n_literals, self.feature_maps)
        clause_state = self.zero_state(n_clauses, self.feature_maps)

        # step_logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        step_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        # unsat_queries = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        # unsat_output = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        last_logits = tf.zeros([n_literals, 1])
        supervised_loss = 0.

        rounds = self.train_rounds if training else self.test_rounds
        cl_adj_matrix = tf.sparse.transpose(adj_matrix)

        for step in tf.range(rounds):
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(literals)
                v1 = tf.concat([literals[:n_vars], literals[n_vars:]], axis=-1)
                v1 = tf.concat([v1, tf.random.normal([n_vars, 4])], axis=-1)
                query = self.literals_query(v1)

                # unsat_queries = unsat_queries.write(step, unsat_clause_count(query, clauses))
                clauses_loss = softplus_loss(query, clauses)
                step_loss = tf.reduce_sum(clauses_loss)

                var_grad = grad_tape.gradient(step_loss, query)
                # TODO: Better way to handle variable and literal size mismatch?
                var_grad = tf.convert_to_tensor(var_grad)
                literals_grad = tf.concat([var_grad[:, :self.query_maps], var_grad[:, self.query_maps:]], axis=0)
                # literals_grad = tf.concat([variables_grad, variables_grad], axis=0)

            if self.use_message_passing:
                clause_messages = tf.sparse.sparse_dense_matmul(cl_adj_matrix, literals)
                clause_unit = tf.concat([clause_state, clause_messages, clauses_loss], axis=-1)
            else:
                clause_unit = tf.concat([clause_state, clauses_loss], axis=-1)

            clause_data = self.clauses_update(clause_unit)

            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_mask, training=training) * 0.25
            clause_state = new_clause_value + 0.1 * clause_state

            literals_loss_all = clause_data[:, 0:self.query_maps]
            literals_loss = tf.sparse.sparse_dense_matmul(adj_matrix, literals_loss_all)

            if self.add_gradient:
                unit = tf.concat([literals, literals_grad, literals_loss], axis=-1)
            else:
                unit = tf.concat([literals, literals_loss], axis=-1)

            new_literals = self.literals_update(unit)
            new_literals = self.variables_norm(new_literals, literals_mask, training=training) * 0.25
            literals = new_literals + 0.1 * literals

            variables = tf.concat([literals[:n_vars], literals[n_vars:]], axis=-1)
            logits = self.literals_output(variables)

            per_clause_loss = softplus_mixed_loss(logits, clauses)
            per_graph_loss = tf.math.segment_sum(per_clause_loss, clauses_mask)
            logit_loss = tf.reduce_sum(tf.sqrt(per_graph_loss + 1e-6))

            step_losses = step_losses.write(step, logit_loss)
            n_unsat_clauses = unsat_clause_count(logits, clauses)
            # unsat_output = unsat_output.write(step, n_unsat_clauses)
            if logit_loss < 0.5 and n_unsat_clauses == 0:
                labels = tf.round(tf.sigmoid(logits))  # now we know the answer, we can use it for supervised training
                supervised_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=last_logits, labels=labels)
                supervised_loss = tf.reduce_mean(supervised_loss)
                last_logits = logits
                break

            last_logits = logits

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            literals = tf.stop_gradient(literals) * 0.2 + literals * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        if training:
            last_clauses = softplus_loss(last_logits, clauses)
            tf.summary.histogram("clauses", last_clauses)
            last_layer_loss = tf.reduce_sum(softplus_log_loss(last_logits, clauses))
            tf.summary.histogram("logits", last_logits)
            tf.summary.scalar("last_layer_loss", last_layer_loss)
            # log_as_histogram("step_losses", step_losses.stack())

            # with tf.name_scope("unsat_clauses"):
                # unsat_q = unsat_queries.stack()
                # unsat_o = unsat_output.stack()
                # log_discreate_as_histogram("queries", unsat_q)
                # log_discreate_as_histogram("outputs", unsat_o)

            tf.summary.scalar("steps_taken", step)
            tf.summary.scalar("supervised_loss", supervised_loss)

            # tf.summary.scalar("alpha_variables", self.alpha_variables)
            # tf.summary.scalar("alpha_clauses", self.alpha_clauses)

            # tf.summary.histogram("residual_variables", tf.sigmoid(self.residual_scale_variables))
            # tf.summary.histogram("residual_clauses", tf.sigmoid(self.residual_scale_clauses))
            # tf.summary.scalar("query_scale", query_scale)
            # tf.summary.scalar("grad_scale", grad_scale)

        return last_logits, tf.reduce_sum(step_losses.stack()) / tf.cast(rounds, tf.float32) + supervised_loss

    @staticmethod
    def flip(literals, n_vars):
        return tf.concat([literals[n_vars:(2 * n_vars), :], literals[0:n_vars, :]], axis=0)

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def train_step(self, adj_matrix, clauses, variable_count, clauses_count):

        with tf.GradientTape() as tape:
            _, loss = self.call(adj_matrix, clauses, variable_count, clauses_count, training=True)
            train_vars = self.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(gradients, train_vars))

        return {
            "loss": loss,
            "gradients": gradients
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_QUERY_MAPS: self.query_maps,
                HP_TRAIN_ROUNDS: self.train_rounds,
                HP_TEST_ROUNDS: self.test_rounds,
                HP_MLP_LAYERS: self.vote_layers,
                }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix, clauses, variable_count, clauses_count):
        predictions, loss = self.call(adj_matrix, clauses, variable_count, clauses_count, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }
