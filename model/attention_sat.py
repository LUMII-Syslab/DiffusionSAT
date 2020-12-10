import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.attention import AdditiveAttention
from layers.layer_normalization import LayerNormalization
from loss.sat import softplus_log_square_loss, unsat_clause_count, softplus_loss
from model.mlp import MLP


class AttentionSAT(Model):

    def __init__(self, optimizer: Optimizer, feature_maps=256, msg_layers=3, vote_layers=3, rounds=32, **kwargs):
        super().__init__(**kwargs, name="AttentionSAT")
        self.rounds = rounds
        self.optimizer = optimizer

        self.literals_mlp = MLP(msg_layers, feature_maps, feature_maps, do_layer_norm=False)
        self.clauses_mlp = MLP(msg_layers, feature_maps, feature_maps, do_layer_norm=False)
        self.variables_query = MLP(msg_layers, feature_maps, feature_maps, do_layer_norm=False)

        self.attention_l = AdditiveAttention(feature_maps, name="attention")
        self.output_layer = MLP(vote_layers, feature_maps * 2, 1, name="output_layer", do_layer_norm=True)
        self.lit_norm = LayerNormalization(axis=-1)

        self.denom = tf.sqrt(tf.cast(feature_maps, tf.float32))
        self.feature_maps = feature_maps

    def zero_state(self, n_units, n_features, stddev=0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        onehot = onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev
        return onehot

    def call(self, adj_matrix, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse adjacency matrix
        n_lits = shape[0]
        n_vars = n_lits // 2

        l_output = self.zero_state(n_lits, self.feature_maps)

        logits = tf.zeros([n_vars, 1])
        step_loss = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)

        for step in tf.range(self.rounds):

            variables = tf.concat([l_output[:n_vars], l_output[n_vars:]], axis=1)  # n_vars x 2
            query = self.variables_query(variables)
            clauses_loss = softplus_loss(query, clauses)

            new_literals = self.attention_l(query=l_output, memory=clauses_loss, adj_matrix=adj_matrix)
            l_output = self.literals_mlp(tf.concat([l_output, new_literals], axis=-1))
            l_output = self.lit_norm(l_output, training=training)

            variables = tf.concat([l_output[:n_vars], l_output[n_vars:]], axis=1)  # n_vars x 2
            logits = self.output_layer(variables)

            loss = softplus_log_square_loss(logits, clauses)
            loss = tf.reduce_sum(loss)
            step_loss = step_loss.write(step, loss)

            n_unsat_clauses = unsat_clause_count(logits, clauses)
            if loss < 0.5 and n_unsat_clauses == 0:
                break

            l_output = tf.stop_gradient(l_output) * 0.2 + l_output * 0.8

        tf.summary.scalar("steps_taken", step)
        return logits, tf.reduce_mean(step_loss.stack())

    @staticmethod
    def flip(literals, n_vars):
        return tf.concat([literals[n_vars:(2 * n_vars), :], literals[0:n_vars, :]], axis=0)

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)])
    def train_step(self, adj_matrix, clauses):
        with tf.GradientTape() as tape:
            logits, loss = self.call(adj_matrix, clauses, training=True)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {
            "loss": loss,
            "gradients": gradients
        }

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)])
    def predict_step(self, adj_matrix, clauses):
        predictions, loss = self.call(adj_matrix, clauses, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }
