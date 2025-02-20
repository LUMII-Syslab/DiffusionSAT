import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense

from data.tsp import PADDING_VALUE
from layers.dense_gnn import DenseGNN
from layers.matrix_se import MatrixSE
from loss.tsp import tsp_loss
from loss.unsupervised_tsp import inverse_identity
from metrics.tsp_metrics import remove_padding, get_unpadded_size
from model.mlp import MLP
from utils.summary import plot_to_image


def inv_sigmoid(y):
    return tf.math.log(y / (1 - y))


class TSPMatrixSE(tf.keras.Model):

    def __init__(self, optimizer, feature_maps=64, block_count=1, rounds=16, use_matrix_se=False, **kwargs):
        super(TSPMatrixSE, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.rounds = rounds
        if use_matrix_se:
            self.graph_layer = MatrixSE(block_count)
        else:
            self.graph_layer = DenseGNN()
        self.input_layer = Dense(feature_maps, activation=None, name="input_layer")
        self.logits_layer = MLP(2, feature_maps, 1, name="logits_layer", do_layer_norm=True, norm_axis=[1, 2])
        n_vertices = 16  # todo: get from somewhere
        self.logit_bias = inv_sigmoid(1.0 / (n_vertices - 1))

    # @tf.function  # only for supervised
    def call(self, inputs, training=None, mask=None, labels=None):
        inputs_norm = inputs * mask * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs * mask), axis=[1, 2], keepdims=True) + 1e-6)
        state = self.input_layer(tf.expand_dims(inputs_norm, -1)) * 0.25
        total_loss = 0.
        input_shape = tf.shape(inputs)
        logits = tf.zeros([input_shape[0], input_shape[1], input_shape[2], 1])
        last_loss = tf.constant(0.0)

        for step in tf.range(self.rounds):
            state = self.graph_layer(state, training=training, mask=mask)
            logits = self.logits_layer(state) + self.logit_bias
            logit_l2 = tf.reduce_mean(tf.square(logits))

            if training:
                loss = tsp_loss(logits, inputs, log_in_tb=training and step == self.rounds - 1, unsupervised=True)
                # loss = tsp_loss(logits, inputs, labels=labels, supervised=True, unsupervised=False)
            else:
                loss = 0.
            total_loss += loss
            last_loss = loss
            # total_loss+=logit_l2 * 1e-4

        if training:
            logit_l2 = tf.reduce_mean(tf.square(logits))
            tf.summary.scalar("L2", logit_l2)
            tf.summary.scalar("last_layer_loss", last_loss)
            tf.summary.histogram("logits", logits)

        return logits, total_loss, last_loss

    def train_step(self, adj_matrix, coords, labels):
        padded_size = tf.shape(adj_matrix)[1]
        mask = tf.cast(tf.not_equal(labels, PADDING_VALUE), tf.float32) * inverse_identity(padded_size)
        with tf.GradientTape() as tape:
            predictions, total_loss, last_loss = self.call(adj_matrix, training=True, mask=mask, labels=labels)
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimize(gradients)

        return {
            "loss": total_loss,
            "gradients": gradients
        }

    @tf.function
    def optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict_step(self, adj_matrix, coords, labels):
        padded_size = tf.shape(adj_matrix)[1]
        mask = tf.cast(tf.not_equal(labels, PADDING_VALUE), tf.float32) * inverse_identity(padded_size)
        predictions, total_loss, last_loss = self.call(adj_matrix, training=False, mask=mask)

        return {
            # "loss": tsp_loss(predictions, adj_matrix, labels=labels),
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def log_visualizations(self, adj_matrix, coords, labels):
        adj_matrix_first = adj_matrix[:1]
        coords_first = coords[:1]
        labels_first = labels[:1]
        padded_size = tf.shape(adj_matrix_first)[1]
        mask = tf.cast(tf.not_equal(labels_first, PADDING_VALUE), tf.float32) * inverse_identity(padded_size)
        predictions, _, _ = self.call(adj_matrix_first, training=False, mask=mask)
        node_count = get_unpadded_size(coords_first[0])
        prediction = remove_padding(predictions[0, :, :, 0], unpadded_size=node_count)
        coord = remove_padding(coords_first[0].numpy(), unpadded_size=node_count)
        label = remove_padding(labels_first[0].numpy(), unpadded_size=node_count)
        figure = self.draw_predictions_and_optimal_path(tf.sigmoid(prediction).numpy(), coord, label)
        tf.summary.image("graph", tf.cast(plot_to_image(figure), tf.float32) / 255.0)

    def draw_predictions_and_optimal_path(self, prediction, coords, label):
        # draws the predicted edges with the opacity corresponding to the model's confidence
        # edges in the optimal path are green, other edges are red
        figure = plt.figure(figsize=(5, 5))
        n = coords.shape[0]

        # draw solution
        for i in range(n):
            for j in range(i):
                larger_edge = max(prediction[i, j], prediction[j, i])
                one_edge = min(prediction[i, j], prediction[j, i])
                ratio = one_edge / larger_edge
                color = (1., 0., ratio)  # red, if one-directional edge, pink if both directions
                alpha = min(1., (prediction[i, j] + prediction[j, i]))
                plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], alpha=alpha, color=color, lw='4')

        # draw label on top
        for i in range(n):
            for j in range(i):
                if label[i, j] == 0.5 or label[j, i] == 0.5 or label[i, j] == 1 or label[j, i] == 1:
                    plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color='black', lw='2',
                             linestyle='--', dashes=(5, 8))

        return figure

    def get_config(self):
        return {}
