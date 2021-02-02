import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.normalization import PairNorm
from loss.sat import softplus_loss, unsat_clause_count, softplus_log_loss, softplus_mixed_loss
from model.mlp import MLP
from utils.parameters_log import *
from utils.summary import log_discreate_as_histogram


class QuerySAT(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=128, msg_layers=3,
                 vote_layers=3, train_rounds=32, test_rounds=64,
                 query_maps=32, supervised=False, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.supervised = supervised
        self.train_rounds = train_rounds
        self.test_rounds = test_rounds
        self.optimizer = optimizer

        self.variables_norm = PairNorm(subtract_mean=True)
        self.clauses_norm = PairNorm(subtract_mean=True)
        self.update_gate = MLP(vote_layers, feature_maps * 2, feature_maps, name="update_gate", do_layer_norm=False)

        self.variables_output = MLP(vote_layers, feature_maps, 1, name="variables_output", do_layer_norm=False)
        self.variables_query = MLP(msg_layers, query_maps * 2, query_maps, name="variables_query", do_layer_norm=False)
        self.clause_mlp = MLP(vote_layers, feature_maps * 3, feature_maps + 1 * query_maps, name="clause_update",
                              do_layer_norm=False)

        self.feature_maps = feature_maps
        self.query_maps = query_maps
        self.vote_layers = vote_layers

    def zero_state(self, n_units, n_features, stddev=0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        onehot = onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev
        return onehot

    def call(self, adj_matrix_pos, adj_matrix_neg, clauses=None, variable_count=None, clauses_count=None, training=None, mask=None, labels=None):
        shape = tf.shape(adj_matrix_pos)
        n_vars = shape[0]
        n_clauses = shape[1]

        graph_count = tf.shape(variable_count)
        graph_id = tf.range(0, graph_count[0])
        variables_mask = tf.repeat(graph_id, variable_count)
        clauses_mask = tf.repeat(graph_id, clauses_count)

        variables = self.zero_state(n_vars, self.feature_maps)
        clause_state = self.zero_state(n_clauses, self.feature_maps)
        step_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        unsat_output = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        last_logits = tf.zeros([n_vars, 1])
        supervised_loss = 0.

        rounds = self.train_rounds if training else self.test_rounds

        for step in tf.range(rounds):
            # make a query for solution, get its value and gradient
            with tf.GradientTape() as grad_tape:
                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1) # add some randomness to avoid zero collapse in normalization
                query = self.variables_query(v1, graph_mask=variables_mask)
                clauses_loss = softplus_loss(query, clauses)
                step_loss = tf.reduce_sum(clauses_loss)
            variables_grad = grad_tape.gradient(step_loss, query)

            # calculate new clause state
            clause_unit = tf.concat([clause_state, clauses_loss], axis=-1)
            clause_data = self.clause_mlp(clause_unit, graph_mask=clauses_mask)
            variables_loss_all = clause_data[:, 0:self.query_maps]
            new_clause_value = clause_data[:, self.query_maps:]
            new_clause_value = self.clauses_norm(new_clause_value, clauses_mask, training=training) * 0.25
            clause_state = new_clause_value + 0.1 * clause_state

            # Aggregate loss over positive edges (x) and negative edges (not x)
            variables_loss_pos = tf.sparse.sparse_dense_matmul(adj_matrix_pos, variables_loss_all)
            variables_loss_neg = tf.sparse.sparse_dense_matmul(adj_matrix_neg, variables_loss_all)

            # calculate new variable state
            unit = tf.concat([variables, variables_grad, variables_loss_pos, variables_loss_neg], axis=-1)
            new_variables = self.update_gate(unit, graph_mask=variables_mask)
            new_variables = self.variables_norm(new_variables, variables_mask, training=training) * 0.25
            variables = new_variables + 0.1 * variables

            # calculate logits and loss
            logits = self.variables_output(variables, graph_mask=variables_mask)
            if self.supervised:
                if labels is not None:
                    logit_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.expand_dims(tf.cast(labels, tf.float32), -1)))
                else: logit_loss = 0.
            else:
                per_clause_loss = softplus_mixed_loss(logits, clauses)
                per_graph_loss = tf.math.segment_sum(per_clause_loss, clauses_mask)
                logit_loss = tf.reduce_sum(tf.sqrt(per_graph_loss + 1e-6))

            step_losses = step_losses.write(step, logit_loss)
            n_unsat_clauses = unsat_clause_count(logits, clauses)
            unsat_output = unsat_output.write(step, n_unsat_clauses)
            if n_unsat_clauses == 0:
                if not self.supervised:
                    labels_got = tf.round(tf.sigmoid(logits))  # now we know the answer, we can use it for supervised training
                    supervised_loss = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=last_logits, labels=labels_got))
                last_logits = logits
                break
            last_logits = logits

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        if training:
            last_clauses = softplus_loss(last_logits, clauses)
            tf.summary.histogram("clauses", last_clauses)
            last_layer_loss = tf.reduce_sum(softplus_log_loss(last_logits, clauses))
            tf.summary.histogram("logits", last_logits)
            tf.summary.scalar("last_layer_loss", last_layer_loss)
            # log_as_histogram("step_losses", step_losses.stack())

            with tf.name_scope("unsat_clauses"):
                unsat_o = unsat_output.stack()
                log_discreate_as_histogram("outputs", unsat_o)

            tf.summary.scalar("steps_taken", step)
            tf.summary.scalar("supervised_loss", supervised_loss)

        return last_logits, tf.reduce_sum(step_losses.stack()) / tf.cast(rounds, tf.float32) + supervised_loss

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)])
    def train_step(self, adj_matrix_pos, adj_matrix_neg, clauses, variable_count, clauses_count, solutions):

        with tf.GradientTape() as tape:
            _, loss = self.call(adj_matrix_pos, adj_matrix_neg, clauses, variable_count, clauses_count, training=True, labels=solutions.flat_values)
            train_vars = self.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(gradients, train_vars))

        return {
            "loss": loss,
            "gradients": gradients
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.TensorSpec(shape=[None], dtype=tf.int32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)
                                  ])
    def predict_step(self, adj_matrix_pos, adj_matrix_neg, clauses, variable_count, clauses_count,solutions):
        predictions, loss = self.call(adj_matrix_pos, adj_matrix_neg, clauses, variable_count, clauses_count,
                                      training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def get_config(self):
        return {HP_MODEL: self.__class__.__name__,
                HP_FEATURE_MAPS: self.feature_maps,
                HP_QUERY_MAPS: self.query_maps,
                HP_TRAIN_ROUNDS: self.train_rounds,
                HP_TEST_ROUNDS: self.test_rounds,
                HP_MLP_LAYERS: self.vote_layers,
                }
