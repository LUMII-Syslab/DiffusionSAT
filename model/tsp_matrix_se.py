from loss.tsp import tsp_loss
from layers.matrix_se import MatrixSE
import tensorflow as tf


class TSPMatrixSE(tf.keras.Model):

    def __init__(self, optimizer, feature_maps=64, output_features=1, block_count=1, **kwargs):
        super(TSPMatrixSE, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.matrix_se = MatrixSE(feature_maps, output_features, block_count)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        return self.matrix_se(inputs, training=training)

    def train_step(self, adj_matrix):
        with tf.GradientTape() as tape:
            predictions = self.call(adj_matrix, training=True)
            loss = tsp_loss(predictions, adj_matrix)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimize(gradients)
        return {
            "loss": loss,
            "gradients": gradients
        }

    @tf.function
    def optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def predict_step(self, adj_matrix):
        predictions = self.call(adj_matrix, training=False)

        return {
            "loss": tsp_loss(predictions, adj_matrix),
            "prediction": tf.squeeze(predictions, axis=-1)
        }
