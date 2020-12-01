import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer

from layers.attention import GraphAttentionLayer
from layers.layer_normalization import LayerNormalization
from loss.sat import softplus_log_square_loss, unsat_clause_count
from model.mlp import MLP
import tensorflow_addons as tfa


class AttentionSAT(Model):

    def __init__(self, optimizer: Optimizer, feature_maps=256, msg_layers=3, vote_layers=3, rounds=16, **kwargs):
        super().__init__(**kwargs, name="AttentionSAT")
        self.rounds = rounds
        self.optimizer = optimizer

        init = tf.initializers.RandomNormal()
        self.L_init = self.add_weight(name="L_init", shape=[1, feature_maps], initializer=init, trainable=True)
        self.C_init = self.add_weight(name="C_init", shape=[1, feature_maps], initializer=init, trainable=True)

        self.literals_mlp = MLP(msg_layers, feature_maps, feature_maps, do_layer_norm=False)
        self.clauses_mlp = MLP(msg_layers, feature_maps, feature_maps, do_layer_norm=False)

        self.attention_l = GraphAttentionLayer(feature_maps, feature_maps, name="attention_l")
        self.attention_c = GraphAttentionLayer(feature_maps, feature_maps, name="attention_c")
        self.layer_norm_3 = LayerNormalization(axis=-1)
        self.layer_norm_4 = LayerNormalization(axis=-1)

        self.output_layer = MLP(vote_layers, feature_maps * 2, 1, name="L_vote", do_layer_norm=False)

        self.denom = tf.sqrt(tf.cast(feature_maps, tf.float32))
        self.feature_maps = feature_maps

    def call(self, adj_matrix, clauses=None, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse adjacency matrix
        n_lits = shape[0]
        n_clauses = shape[1]
        n_vars = n_lits // 2

        l_output = tf.tile(self.L_init / self.denom, [n_lits, 1])
        c_output = tf.tile(self.C_init / self.denom, [n_clauses, 1])
        logits = tf.zeros([n_vars, 1])

        #
        # l_output = tf.random.truncated_normal([n_lits, self.feature_maps], stddev=0.025)
        # c_output = tf.random.truncated_normal([n_clauses, self.feature_maps], stddev=0.025)

        step_loss = tf.TensorArray(tf.float32, size=self.rounds, clear_after_read=True)

        for step in tf.range(self.rounds):
            new_clauses = self.attention_c(c_output, l_output, tf.sparse.transpose(adj_matrix))
            c_output = tf.concat([c_output, new_clauses], axis=-1)
            new_clauses2 = self.clauses_mlp(c_output, training=training)
            c_output = self.layer_norm_4(new_clauses2)

            new_literals = self.attention_l(l_output, c_output, adj_matrix)
            l_output = tf.concat([l_output, self.flip(new_literals, n_vars)], axis=-1)
            new_literals2 = self.literals_mlp(l_output, training=training)
            l_output = self.layer_norm_3(new_literals2)

            variables = tf.concat([l_output[:n_vars], l_output[n_vars:]], axis=1)  # n_vars x 2
            logits = self.output_layer(variables)

            loss = softplus_log_square_loss(logits, clauses)
            loss = tf.reduce_sum(loss)
            step_loss = step_loss.write(step, loss)

            n_unsat_clauses = unsat_clause_count(logits, clauses)
            if loss < 0.5 and n_unsat_clauses == 0:
                break

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
