from loss.tsp import tsp_loss
from layers.matrix_se import MatrixSE
import tensorflow as tf
from tensorflow.keras.layers import Dense


class TSPMatrixSE(tf.keras.Model):

    def __init__(self, optimizer, feature_maps = 64, block_count = 1, **kwargs):
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
