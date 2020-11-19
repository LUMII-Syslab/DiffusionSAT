import tensorflow as tf
from tensorflow.keras.models import Model

from layers.layer_normalization import LayerNormalization
from loss.sat import softplus_loss, softplus_log_square_loss
from model.mlp import MLP
from utils.summary import log_as_histogram


class QuerySAT(Model):

    def __init__(self, feature_maps=256, msg_layers=3, vote_layers=3, rounds=16, query_maps = 64, **kwargs):
        super().__init__(**kwargs, name="QuerySAT")
        self.rounds = rounds

        self.variables_norm = LayerNormalization(axis=-1)
        self.clauses_norm = LayerNormalization(axis=-1)

        self.update_gate = MLP(vote_layers, feature_maps * 2, feature_maps, name="update_gate")

        # self.forget_gate = MLP(msg_layers, feature_maps, feature_maps,
        #                        out_activation=tf.sigmoid,
        #                        out_bias = -1,
        #                        name="forget_gate")

        self.variables_output = MLP(vote_layers, feature_maps, 1, name="variables_output")
        self.variables_query = MLP(msg_layers, query_maps * 2, query_maps, name="variables_query")
        #self.clause_update = MLP(vote_layers, feature_maps * 2, feature_maps, name="clause_update")
        #self.clause_update_gate = MLP(vote_layers, feature_maps, feature_maps, out_activation = tf.sigmoid, out_bias = -1, name="clause_update_gate")
        #self.query_pos_inter = MLP(msg_layers, query_maps * 2, query_maps, name="query_pos_inter")
        #self.query_neg_inter = MLP(msg_layers, query_maps * 2, query_maps, name="query_neg_inter")
        self.clause_mlp = MLP(vote_layers, feature_maps * 3, feature_maps + 2*query_maps, name="clause_update")

        self.feature_maps = feature_maps
        self.query_maps = query_maps

    def zero_state(self, n_units, n_features, stddev = 0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        onehot = onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev
        return onehot

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32),
                                  tf.TensorSpec(shape=(), dtype=tf.bool)])
    def call(self, adj_matrix_pos, adj_matrix_neg, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix_pos)
        n_vars = shape[0]
        n_clauses = shape[1]

        #variables = tf.random.truncated_normal([n_vars, self.feature_maps], stddev=0.25)
        variables = self.zero_state(n_vars, self.feature_maps)
        clause_state = self.zero_state(n_clauses, self.feature_maps)
        step_logits = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)
        step_losses = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)

        for step in tf.range(self.rounds):
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(variables)
                #v1 = tf.concat([variables, tf.random.normal([n_vars, self.query_maps])], axis=-1)
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
            clause_unit = tf.concat([clause_state, clauses_loss], axis=-1)
            clause_data = self.clause_mlp(clause_unit)
            variables_loss_pos = clause_data[:,0:self.query_maps]
            variables_loss_neg = clause_data[:, self.query_maps:2*self.query_maps]
            new_clause_value = clause_data[:, 2 * self.query_maps:]
            #variables_loss_pos = self.query_pos_inter(clause_unit)
            variables_loss_pos = tf.sparse.sparse_dense_matmul(adj_matrix_pos, variables_loss_pos)

            # Aggregate loss over negative edges (not x)
            #variables_loss_neg = self.query_neg_inter(clause_unit)
            variables_loss_neg = tf.sparse.sparse_dense_matmul(adj_matrix_neg, variables_loss_neg)
            #new_clause_value = self.clause_update(clause_unit)
            new_clause_value = self.clauses_norm(new_clause_value, training=training)*0.25
            #new_clause_gate = self.clause_update_gate(clause_unit)
            #tf.summary.histogram("clause_gate" + str(step), new_clause_gate)
            #clause_state = (1 - new_clause_gate) * clause_state + new_clause_gate * new_clause_value
            clause_state = new_clause_value# + 0.1*clause_state


            unit = tf.concat([variables, variables_grad, variables_loss_pos, variables_loss_neg], axis=-1)

            #forget_gate = self.forget_gate(unit)
            new_variables = self.update_gate(unit)
            new_variables = self.variables_norm(new_variables, training=training)*0.25  # TODO: Rethink normalization
            #tf.summary.histogram("gate" + str(step), forget_gate)

            #variables = (1 - forget_gate) * variables + forget_gate * new_variables
            variables = new_variables# + 0.1*variables

            logits = self.variables_output(variables)
            step_logits = step_logits.write(step, logits)
            step_losses = step_losses.write(step, tf.reduce_sum(softplus_log_square_loss(logits, clauses)))

            # due to the loss at each level, gradients accumulate on the backward pass and may become very large for the first layers
            # reduce the gradient magnitude to remedy this
            variables = tf.stop_gradient(variables) * 0.2 + variables * 0.8
            clause_state = tf.stop_gradient(clause_state) * 0.2 + clause_state * 0.8

        step_logits_tensor = step_logits.stack()  # step_count x literal_count
        last_layer_loss = tf.reduce_sum(softplus_log_square_loss(step_logits_tensor[-1], clauses))
        tf.summary.scalar("last_layer_loss", last_layer_loss)
        log_as_histogram("step_losses", step_losses.stack())
        return step_logits_tensor
