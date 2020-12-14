import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.attention import AdditiveAttention
from layers.normalization import LayerNormalization
from loss.sat import unsat_clause_count, softplus_loss, softplus_log_loss
from model.mlp import MLP


class AttentionSAT(Model):

    def __init__(self, optimizer: Optimizer, feature_maps=256, msg_layers=3, vote_layers=3, rounds=16, **kwargs):
        super().__init__(**kwargs, name="AttentionSAT")
        self.rounds = rounds
        self.optimizer = optimizer

        self.literals_mlp = MLP(msg_layers, feature_maps, feature_maps, do_layer_norm=True)
        self.variables_query = MLP(msg_layers, feature_maps, 64, do_layer_norm=True)

        self.attention_l = AdditiveAttention(feature_maps, name="attention")
        self.output_layer = MLP(vote_layers, feature_maps, 1, name="output_layer", do_layer_norm=True)
        self.lit_norm = LayerNormalization(axis=0, subtract_mean=True)

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

        supervised_loss = 0.
        logits = tf.zeros([n_vars, 1])
        step_loss = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)

        for step in tf.range(self.rounds):
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(l_output)
                lits = tf.concat([l_output, tf.random.normal([n_lits, 4])], axis=-1)
                variables = tf.concat([lits[:n_vars], lits[n_vars:]], axis=1)  # n_vars x 2
                query = self.variables_query(variables)
                clauses_loss = softplus_loss(query, clauses)
                total_loss = tf.reduce_sum(clauses_loss)
            literals_grad = grad_tape.gradient(total_loss, query)
            literals_grad = tf.concat(tf.split(literals_grad, 2, axis=1), axis=0)

            literals_loss = tf.sparse.sparse_dense_matmul(adj_matrix, clauses_loss)
            literals_unit = tf.concat([l_output, literals_grad, literals_loss], axis=-1)

            clauses_gradient = tf.sparse.sparse_dense_matmul(adj_matrix, literals_grad, adjoint_a=True)
            clauses_full = tf.sparse.sparse_dense_matmul(adj_matrix, l_output, adjoint_a=True)
            clauses_unit = tf.concat([clauses_full, clauses_gradient, clauses_loss], axis=-1)

            new_literals = self.attention_l(query=literals_unit, memory=clauses_unit, adj_matrix=adj_matrix)

            tf.summary.histogram("aggregated_loss", new_literals)

            l_output = self.literals_mlp(tf.concat([literals_unit, self.flip(new_literals, n_lits)], axis=-1), training=training)

            tf.summary.histogram("new_literals", l_output)

            l_output = self.lit_norm(l_output, training=training)
            tf.summary.histogram("literals", l_output)

            variables = tf.concat([l_output[:n_vars], l_output[n_vars:]], axis=1)  # n_vars x 2
            logits = self.output_layer(variables, training=training) * 0.25

            loss = softplus_log_loss(logits, clauses)
            loss = tf.reduce_sum(loss)
            step_loss = step_loss.write(step, loss)

            n_unsat_clauses = unsat_clause_count(logits, clauses)
            if loss < 0.5 and n_unsat_clauses == 0:
                labels = tf.round(tf.sigmoid(logits))  # now we know the answer, we can use it for supervised training
                supervised_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                break

            l_output = tf.stop_gradient(l_output) * 0.2 + l_output * 0.8

        tf.summary.scalar("steps_taken", step)
        tf.summary.histogram("logits", logits)
        return logits, tf.reduce_mean(step_loss.stack()) + supervised_loss

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
