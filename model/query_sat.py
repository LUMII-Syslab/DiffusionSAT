import tensorflow as tf
from tensorflow.keras.models import Model

from layers.layer_normalization import LayerNormalization
from loss.sat import softplus_loss
from model.mlp import MLP


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
                                  tf.TensorSpec(shape=(), dtype=bool)])
    def call(self, adj_matrix_pos, adj_matrix_neg, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix_pos)
        n_vars = shape[0]

        variables = tf.random.truncated_normal([n_vars, self.feature_maps], stddev=0.25)
        step_logits = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)

        for step in tf.range(self.rounds):
            query = self.variables_query(variables)
            clauses_loss = softplus_loss(query, clauses)

            variables_loss_pos = self.query_pos_inter(clauses_loss)
            variables_loss_pos = tf.sparse.sparse_dense_matmul(adj_matrix_pos, variables_loss_pos)

            variables_loss_neg = self.query_neg_inter(clauses_loss)
            variables_loss_neg = tf.sparse.sparse_dense_matmul(adj_matrix_neg, variables_loss_neg)

            unit = tf.concat([variables, variables_loss_pos, variables_loss_neg], axis=-1)

            forget_gate = self.forget_gate(unit)
            new_variables = self.update_gate(unit)

            variables = (1 - forget_gate) * variables + forget_gate * new_variables
            variables = self.variables_norm(variables, training=training)  # TODO: Rethink normalization

            query = self.variables_output(variables)
            step_logits = step_logits.write(step, query)

        return step_logits.stack()  # step_count x literal_count x 1
