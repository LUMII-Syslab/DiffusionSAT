import tensorflow as tf
from tensorflow.keras.models import Model

from layers.layer_normalization import LayerNormalization
from loss.sat import softplus_loss, softplus_log_square_loss
from model.mlp import MLP
from utils.summary import log_as_histogram


class QuerySAT(Model):

    def __init__(self, feature_maps=512, msg_layers=3, vote_layers=3, rounds=16, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.rounds = rounds

        self.variables_norm = LayerNormalization(axis=-1)

        self.update_gate = MLP(msg_layers, feature_maps, feature_maps, name="update_gate")

        self.forget_gate = MLP(msg_layers, feature_maps, feature_maps,
                               out_activation=tf.sigmoid,
                               name="forget_gate")

        self.variables_output = MLP(vote_layers, feature_maps, 1, name="variables_output")
        self.variables_query = MLP(vote_layers, feature_maps, feature_maps, name="variables_query")
        self.query_pos_inter = MLP(vote_layers, feature_maps, feature_maps // 2, name="query_pos_inter")
        self.query_neg_inter = MLP(vote_layers, feature_maps, feature_maps // 2, name="query_neg_inter")

        self.feature_maps = feature_maps

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self, adj_matrix_pos, adj_matrix_neg, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix_pos)
        n_vars = shape[0]

        variables = tf.random.truncated_normal([n_vars, self.feature_maps], stddev=0.25)
        step_logits = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)
        step_losses = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)

        for step in tf.range(self.rounds):
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(variables)
                query = self.variables_query(variables)
                clauses_loss = softplus_loss(query, clauses)
                step_loss = tf.reduce_sum(clauses_loss)
            variables_grad = grad_tape.gradient(step_loss, query)
            # logit_grad = grad_tape.gradient(step_loss, logits)
            # var_grad = self.grad2var(logit_grad)
            # literal_grad = tf.concat([var_grad[:, self.feature_maps:],var_grad[:, 0:self.feature_maps]], axis=0)
            # tf.summary.histogram("lit_grad"+str(r), literal_grad)
            # tf.summary.histogram("logit_grad" + str(r), logit_grad)

            # Aggregate loss over positive edges (x)
            variables_loss_pos = self.query_pos_inter(clauses_loss)
            variables_loss_pos = tf.sparse.sparse_dense_matmul(adj_matrix_pos, variables_loss_pos)

            # Aggregate loss over negative edges (not x)
            variables_loss_neg = self.query_neg_inter(clauses_loss)
            variables_loss_neg = tf.sparse.sparse_dense_matmul(adj_matrix_neg, variables_loss_neg)

            unit = tf.concat([variables, variables_grad, variables_loss_pos, variables_loss_neg], axis=-1)

            forget_gate = self.forget_gate(unit)
            new_variables = self.update_gate(unit)  # TODO: Try GRRUA by @Ronalds
            new_variables = self.variables_norm(new_variables, training=training)  # TODO: Rethink normalization

            variables = (1 - forget_gate) * variables + forget_gate * new_variables

            logits = self.variables_output(variables)
            step_logits = step_logits.write(step, logits)
            step_losses = step_losses.write(step, tf.reduce_sum(softplus_log_square_loss(logits, clauses)))

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8

        step_logits_tensor = step_logits.stack()  # step_count x literal_count
        last_layer_loss = tf.reduce_sum(softplus_log_square_loss(step_logits_tensor[-1], clauses))
        tf.summary.scalar("last_layer_loss", last_layer_loss)
        log_as_histogram("step_losses", step_losses.stack())
        return step_logits_tensor
