from enum import Enum

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

import utils.shuffle as shuffle_utils
from layers.normalization import LayerNormalization
from loss.sat import softplus_mixed_loss, softplus_loss
from model.mlp import MLP


class ShuffleGNN(tf.keras.models.Model):

    def __init__(self, optimizer, block_count=2, feature_maps=128, **kwargs):
        super(ShuffleGNN, self).__init__(**kwargs)
        self.feature_maps = feature_maps
        self.benes_block = BenesBlock()
        self.L_vote = MLP(3, 128, 1, name="L_vote", do_layer_norm=False)
        self.variables_query = MLP(2, 128, 32, name="clauses_mlp")
        self.loss_interp = MLP(2, 128, 128, name="loss_interpretation")
        self.press = Dense(128)
        self.optimizer = optimizer

    def zero_state(self, n_units, n_features, stddev=0.25):
        onehot = tf.one_hot(tf.zeros([n_units], dtype=tf.int64), n_features)
        onehot -= 1 / n_features
        onehot = onehot * tf.sqrt(tf.cast(n_features, tf.float32)) * stddev
        return onehot

    def call(self, adj_matrix, clauses, training=None, mask=None):
        shape = tf.shape(adj_matrix)  # inputs is sparse adjacency matrix
        n_lits = shape[0]
        n_clauses = shape[1]
        n_vars = n_lits // 2

        literals = self.zero_state(n_lits, self.feature_maps)

        total_loss = 0.
        logits = tf.zeros([n_vars, 1])
        hidden = literals

        steps = 4 if training else 16

        for _ in tf.range(steps):
            loss, grad = self.calc_loss(hidden, adj_matrix, clauses)
            hidden = tf.concat([hidden, loss, grad], axis=-1)
            hidden = self.press(hidden)
            hidden = self.benes_block(hidden, adj_matrix=adj_matrix, clauses=clauses, training=training)

            variables = tf.concat([hidden[:n_vars], hidden[n_vars:]], axis=1)  # n_vars x 2
            logits = self.L_vote(variables)
            total_loss += tf.reduce_sum(softplus_mixed_loss(logits, clauses))

            hidden = tf.stop_gradient(hidden) * 0.2 + hidden * 0.8

        return logits, total_loss / 8.0

    def calc_loss(self, inputs, adj_matrix, clauses):
        length = tf.shape(inputs)[0]

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(inputs)
            v1 = tf.concat([inputs, tf.random.normal([length, 4])], axis=-1)
            v1 = tf.concat([v1[:length // 2], v1[length // 2:]], axis=1)  # n_vars x 2
            query = self.variables_query(v1)
            clauses_loss = softplus_loss(query, clauses)
            step_loss = tf.reduce_sum(clauses_loss)
        variables_grad = grad_tape.gradient(step_loss, query)
        literals_grad = tf.concat([variables_grad, variables_grad], axis=0)

        clauses_loss = self.loss_interp(clauses_loss)
        clauses_loss = tf.sparse.sparse_dense_matmul(adj_matrix, clauses_loss)

        return clauses_loss, literals_grad

    @tf.function(input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL
                 )
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
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def predict_step(self, adj_matrix, clauses):
        predictions, loss = self.call(adj_matrix, clauses, training=False)

        return {
            "loss": loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }


class SwitchUnit(tf.keras.layers.Layer):

    def __init__(self, name, channel_count=2, dropout_rate=0.0, **kwargs):
        super(SwitchUnit, self).__init__(name=name, **kwargs)
        self.channel_count = channel_count
        self.dropout_rate = dropout_rate
        self.residual_weight = 0.9
        self.candidate_weight = np.sqrt(1 - self.residual_weight ** 2) * 0.25
        self.scale_init = np.log(self.residual_weight / (1 - self.residual_weight))

        self.num_units = None
        self.reshaped_units = None
        self.residual_scale = None
        self.layer_norm = None
        self.dropout = None

        self.linear_one = None
        self.linear_two = None

    def build(self, input_shape):
        self.num_units = input_shape.as_list()[1]
        self.reshaped_units = self.num_units * self.channel_count

        initializer = tf.constant_initializer(self.scale_init)
        self.residual_scale = self.add_weight("residual", [self.num_units], initializer=initializer)

        self.linear_one = Dense(self.reshaped_units * 2, name="linear_one", use_bias=False)
        self.linear_two = Dense(self.reshaped_units, name="linear_two")
        # self.variables_query = MLP(2, self.num_units, 32, name="clauses_mlp")
        # self.loss_interp = MLP(2, self.num_units, self.num_units, name="loss_interpretation")

        self.layer_norm = LayerNormalization(axis=0, subtract_mean=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, adj_matrix=None, clauses=None, training=False, **kwargs):
        shape = tf.shape(inputs)
        length, num_units = shape[0], shape[1]

        # with tf.GradientTape() as grad_tape:
        #     grad_tape.watch(inputs)
        #     v1 = tf.concat([inputs, tf.random.normal([length, 4])], axis=-1)
        #     v1 = tf.concat([v1[:length // 2], v1[length // 2:]], axis=1)  # n_vars x 2
        #     query = self.variables_query(v1)
        #     clauses_loss = softplus_loss(query, clauses)
        #     step_loss = tf.reduce_sum(clauses_loss)
        # variables_grad = grad_tape.gradient(step_loss, query)
        # literals_grad = tf.concat([variables_grad, variables_grad], axis=0)
        #
        # clauses_loss = self.loss_interp(clauses_loss)
        # clauses_loss = tf.sparse.sparse_dense_matmul(adj_matrix, clauses_loss)
        #
        # variables = tf.concat([inputs, literals_grad, clauses_loss], axis=-1)
        variables = tf.reshape(inputs, shape=[length // self.channel_count, self.channel_count * self.num_units])

        first_linear = self.linear_one(variables)
        norm = self.layer_norm(first_linear)
        gelu = tf.nn.leaky_relu(norm)
        candidate = self.linear_two(gelu)
        candidate = tf.reshape(candidate, [length, -1])

        residual_scale = tf.nn.sigmoid(self.residual_scale)
        return residual_scale * inputs + self.flip(candidate, length) * self.candidate_weight

    @staticmethod
    def flip(literals, n_vars):
        return tf.concat([literals[n_vars:(2 * n_vars), :], literals[0:n_vars, :]], axis=0)


class ShuffleType(Enum):
    LEFT = shuffle_utils.rol
    RIGHT = shuffle_utils.ror

    def __call__(self, *args, **kwargs):
        self.value(*args)


class ShuffleLayer(tf.keras.layers.Layer):
    """ Implements left quaternary cyclic shift for input tensor as described in
     "Two-Dimensional Benes Network" by Uh-Sock Rhee and Mir M. Mirsalehi
    """

    def __init__(self, do_ror=False, **kwargs):
        super(ShuffleLayer, self).__init__(trainable=False, **kwargs)
        self.do_ror = do_ror

    def call(self, inputs, **kwargs):
        """Shuffles the elements according to bitwise left or right rotation on their indices"""
        length = tf.shape(inputs)[0]

        rev_indices_rol_0 = (tf.range((length + 1) // 2) * 2) % length
        rev_indices_rol_1 = (tf.range(length // 2) * 2 + 1) % length
        rev_indices_rol = tf.concat([rev_indices_rol_0, rev_indices_rol_1], axis=0)
        rev_indices_ror = tf.math.invert_permutation(rev_indices_rol)

        if self.do_ror:
            rev_indices = rev_indices_ror
        else:
            rev_indices = rev_indices_rol

        return tf.gather(inputs, rev_indices, axis=0)


class BenesBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward = SwitchUnit("forward", dropout_rate=0.0)
        self.reverse = SwitchUnit("reverse", dropout_rate=0.0)
        self.middle = SwitchUnit("middle", dropout_rate=0.0)
        self.shuffle_forward = ShuffleLayer(do_ror=True)
        self.shuffle_reverse = ShuffleLayer(do_ror=False)

    def call(self, inputs, adj_matrix=None, clauses=None, training=False, **kwargs):
        side = tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
        bit_length = tf.math.log(side - 1) / tf.math.log(2.)
        level_count = tf.cast(tf.math.floor(bit_length), tf.int32)

        last_layer = inputs
        for _ in tf.range(level_count):
            last_layer = self.forward(last_layer, adj_matrix=adj_matrix, clauses=clauses, training=training)
            last_layer = self.shuffle_forward(last_layer)

        for _ in tf.range(level_count):
            last_layer = self.reverse(last_layer, adj_matrix=adj_matrix, clauses=clauses, training=training)
            last_layer = self.shuffle_reverse(last_layer)

        last_layer = self.middle(last_layer, adj_matrix=adj_matrix, clauses=clauses, training=training)

        return last_layer
