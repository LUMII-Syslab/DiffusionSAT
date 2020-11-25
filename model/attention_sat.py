import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.attention import GraphAttentionLayer
from layers.layer_normalization import LayerNormalization
from loss.sat import softplus_loss, softplus_log_square_loss, unsat_clause_count
from model.mlp import MLP


class AttentionSAT(Model):

    def __init__(self, optimizer: Optimizer,
                 feature_maps=256, msg_layers=3,
                 vote_layers=3, rounds=32,
                 query_maps=64, **kwargs):
        super().__init__(**kwargs, name="AttentionSAT")
        self.rounds = rounds
        self.optimizer = optimizer

        self.variables_norm = LayerNormalization(axis=-1)
        self.clauses_norm = LayerNormalization(axis=-1)

        self.update_gate = MLP(vote_layers, feature_maps * 2, feature_maps, name="update_gate")
        self.variables_output = MLP(vote_layers, feature_maps, 1, name="variables_output")
        self.variables_query = MLP(msg_layers, query_maps * 2, query_maps, name="variables_query")
        self.clause_pos_mlp = GraphAttentionLayer(feature_maps * 2, feature_maps, name="clause_update_pos")
        self.clause_neg_mlp = GraphAttentionLayer(feature_maps * 2, feature_maps, name="clause_update_neg")

        self.feature_maps = feature_maps
        self.query_maps = query_maps

    def zero_state(self, n_units, n_features, stddev=0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        onehot = onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev
        return onehot

    def call(self, adj_matrix_pos, adj_matrix_neg, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix_pos)
        n_vars = shape[0]
        n_clauses = shape[1]

        variables = self.zero_state(n_vars, self.feature_maps)
        clause_state = self.zero_state(n_clauses, self.feature_maps)
        step_logits = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)
        step_losses = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=True)

        for step in tf.range(self.rounds):
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(variables)
                v1 = tf.concat([variables, tf.random.normal([n_vars, 4])], axis=-1)
                query = self.variables_query(v1)
                clauses_loss = softplus_loss(query, clauses)
                step_loss = tf.reduce_sum(clauses_loss)
            variables_grad = grad_tape.gradient(step_loss, query)

            # Aggregate loss over positive edges (x)
            variables_loss_pos = self.clause_pos_mlp(variables, clauses_loss, adj_matrix_pos)

            # Aggregate loss over negative edges (not x)
            variables_loss_neg = self.clause_neg_mlp(variables, clauses_loss, adj_matrix_neg)

            unit = tf.concat([variables, variables_grad, variables_loss_pos, variables_loss_neg], axis=-1)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, training=training) * 0.25  # TODO: Rethink normalization

            variables = new_variables

            logits = self.variables_output(variables)

            step_logits = step_logits.write(step, logits)
            logit_loss = tf.reduce_sum(softplus_log_square_loss(logits, clauses))
            step_losses = step_losses.write(step, logit_loss)
            n_unsat_clauses = unsat_clause_count(logits, clauses)
            if logit_loss < 0.5 and n_unsat_clauses == 0:
                break

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        step_logits_tensor = step_logits.stack()  # step_count x literal_count
        last_layer_loss = tf.reduce_sum(softplus_log_square_loss(step_logits_tensor[-1], clauses))
        tf.summary.scalar("last_layer_loss", last_layer_loss)
        # log_as_histogram("step_losses", step_losses.stack())
        tf.summary.scalar("steps_taken", step)

        return step_logits_tensor[-1], tf.reduce_mean(step_losses.stack())

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)])
    def train_step(self, adj_matrix_pos, adj_matrix_neg, clauses):

        with tf.GradientTape() as tape:
            _, loss = self.call(adj_matrix_pos, adj_matrix_neg, clauses, training=True)
            train_vars = self.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(gradients, train_vars))

        return {
            "loss": loss,
            "gradients": gradients
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)])
    def predict_step(self, adj_matrix_pos, adj_matrix_neg, clauses):
        predictions, loss = self.call(adj_matrix_pos, adj_matrix_neg, clauses, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }
