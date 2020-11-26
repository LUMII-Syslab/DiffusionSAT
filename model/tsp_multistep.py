from loss.unsupervised_tsp import tsp_unsupervised_loss
from layers.matrix_se import MatrixSE
import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from utils.summary import plot_to_image


def inv_sigmoid(y):
    return tf.math.log(y / (1 - y))


class MultistepTSP(tf.keras.Model):

    def __init__(self, optimizer, feature_maps = 64, block_count = 1, rounds = 1, **kwargs):
        super(MultistepTSP, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.matrix_se = MatrixSE(block_count)
        self.input_layer = Dense(feature_maps, activation=None, name="input_layer")
        self.logits_layer = Dense(1, activation=None, name="logits_layer")
        self.rounds = rounds
        n_vertices = 8 #todo: get from somewhere
        self.logit_bias = inv_sigmoid(1.0/(n_vertices-1))

    def draw_graph(self, x, coords):
        figure = plt.figure(figsize=(5, 5))
        n = coords.shape[0]
        for i in range(n):
            for j in range(i):
                both_edges = max(x[i][j], x[j][i])
                one_edge = min(x[i][j], x[j][i])
                ratio = one_edge / both_edges
                color = (1., 0., ratio)  # red, if one-directional edge, pink if both directions
                plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], alpha=both_edges, color=color, lw='3')
        return figure

    def call(self, inputs, training=None, mask=None):
        inputs_norm = inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=[1,2], keepdims=True)+1e-6)
        state = self.input_layer(tf.expand_dims(inputs_norm,-1))*0.25
        total_loss = 0
        logits = None
        last_loss = None

        for step in tf.range(self.rounds):
            state = self.matrix_se(state, training=training)
            logits = self.logits_layer(state)+self.logit_bias
            loss = tsp_unsupervised_loss(logits, inputs_norm, log_in_tb = training and step==self.rounds-1)
            total_loss += loss
            last_loss = loss

        return logits, total_loss, last_loss

    def train_step(self, adj_matrix, coords):
        with tf.GradientTape() as tape:
            predictions, total_loss, last_loss = self.call(adj_matrix, training=True)
            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimize(gradients)
        return {
            "loss": last_loss,
            "gradients": gradients
        }

    @tf.function
    def optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict_step(self, adj_matrix, coords):
        predictions, total_loss, last_loss = self.call(adj_matrix, training=False)
        figure = self.draw_graph(tf.sigmoid(predictions[0, :, :, 0]).numpy(), coords[0].numpy())
        tf.summary.image("graph", tf.cast(plot_to_image(figure), tf.float32) / 255.0)

        return {
            "loss": last_loss,
            "prediction": tf.squeeze(predictions, axis=-1)
        }
