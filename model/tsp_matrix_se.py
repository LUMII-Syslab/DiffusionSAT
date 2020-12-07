import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

from loss.tsp import tsp_loss
from layers.matrix_se import MatrixSE
from utils.summary import plot_to_image
from data.tsp import remove_padding, get_unpadded_size
from layers.dense_gnn import DenseGNN

def inv_sigmoid(y):
    return tf.math.log(y / (1 - y))

class TSPMatrixSE(tf.keras.Model):

    def __init__(self, optimizer, feature_maps=64, block_count=1, rounds=4, use_matrix_se = False, **kwargs):
        super(TSPMatrixSE, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.rounds = rounds
        if use_matrix_se:
            self.graph_layer = MatrixSE(block_count)
        else:
            self.graph_layer = DenseGNN()
        self.input_layer = Dense(feature_maps, activation=None, name="input_layer")
        self.logits_layer = Dense(1, activation=None, name="logits_layer")
        n_vertices = 16  # todo: get from somewhere
        self.logit_bias = inv_sigmoid(1.0 / (n_vertices - 1))


    #@tf.function # only for supervised
    def call(self, inputs, training=None, mask=None, labels = None):
        inputs_norm = inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis=[1, 2], keepdims=True)+1e-6) #todo: masked norm
        state = self.input_layer(tf.expand_dims(inputs_norm, -1))*0.25
        total_loss = 0.
        logits = None
        last_loss = None

        for step in tf.range(self.rounds):
            state = self.graph_layer(state, training=training)
            logits = self.logits_layer(state)+self.logit_bias
            if training:
                loss = tsp_loss(logits, inputs, log_in_tb=training and step == self.rounds - 1, unsupervised=True)
                #loss = tsp_loss(logits, inputs, labels=labels, supervised=True, unsupervised=True)
            else: loss = 0.
            total_loss += loss
            last_loss = loss

        return logits, total_loss, last_loss

    def train_step(self, adj_matrix, coords, labels):
        with tf.GradientTape() as tape:
            predictions, total_loss, last_loss = self.call(adj_matrix, training=True, labels=labels)
            gradients = tape.gradient(total_loss, self.trainable_variables)
            self.optimize(gradients)
        return {
            "loss": last_loss,
            "gradients": gradients
        }

    @tf.function
    def optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict_step(self, adj_matrix, coords, labels):
        predictions, total_loss, last_loss = self.call(adj_matrix, training=False)

        return {
            # "loss": tsp_loss(predictions, adj_matrix, labels=labels),
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def log_visualizations(self, adj_matrix, coords, labels):
        predictions, _, _ = self.call(adj_matrix, training=False)
        node_count = get_unpadded_size(coords[0])
        prediction = remove_padding(predictions[0, :, :, 0], unpadded_size=node_count)
        coord = remove_padding(coords[0].numpy(), unpadded_size=node_count)
        label = remove_padding(labels[0].numpy(), unpadded_size=node_count)
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
                one_edge = min(prediction[i,j], prediction[j,i])
                ratio = one_edge / larger_edge
                color = (1., 0., ratio)  # red, if one-directional edge, pink if both directions
                alpha = min(1., (prediction[i, j] + prediction[j, i]))
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], alpha=alpha, color=color, lw='4')

        # draw label on top
        for i in range(n):
            for j in range(i):
                if label[i,j] == 0.5 or label[j,i] == 0.5:
                    plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], color='black', lw='2', linestyle='--', dashes=(5, 8))

        return figure
