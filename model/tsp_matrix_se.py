import tensorflow as tf
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

from loss.tsp import tsp_loss
from layers.matrix_se import MatrixSE
from utils.summary import plot_to_image
from data.tsp import remove_padding, get_unpadded_size


class TSPMatrixSE(tf.keras.Model):

    def __init__(self, optimizer, feature_maps=64, block_count=1, **kwargs):
        super(TSPMatrixSE, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.matrix_se = MatrixSE(block_count)
        self.input_layer = Dense(feature_maps, activation=None, name="input_layer")
        self.logits_layer = Dense(1, activation=None, name="logits_layer")


    @tf.function
    def call(self, inputs, training=None, mask=None):
        state = self.input_layer(tf.expand_dims(inputs, -1))
        state = self.matrix_se(state, training=training)
        logits = self.logits_layer(state)
        return logits

    def train_step(self, adj_matrix, coords, labels):
        with tf.GradientTape() as tape:
            predictions = self.call(adj_matrix, training=True)
            loss = tsp_loss(predictions, adj_matrix, labels=labels, supervised=True, unsupervised=False)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimize(gradients)
        return {
            "loss": loss,
            "gradients": gradients
        }

    @tf.function
    def optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict_step(self, adj_matrix, coords, labels):
        predictions = self.call(adj_matrix, training=False)

        return {
            # "loss": tsp_loss(predictions, adj_matrix, labels=labels),
            "prediction": tf.squeeze(predictions, axis=-1)
        }

    def log_visualizations(self, adj_matrix, coords, labels):
        predictions = self.call(adj_matrix, training=False)
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
        for i in range(n):
            for j in range(i):
                larger_edge = max(prediction[i][j], prediction[j][i])
                if label[i][j] == 1:
                    color = (0., 1., 0.)  # green if optimal edge
                else:
                    color = (1., 0., 0.)  # red if non-optimal edge
                plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], alpha=larger_edge, color=color, lw='3')
        return figure
