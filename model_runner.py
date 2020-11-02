import tensorflow as tf


class ModelRunner:

    def __init__(self,
                 model_fn: tf.keras.Model,
                 loss_fn: callable,
                 optimizer: tf.optimizers.Optimizer) -> None:
        self.model = model_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(features, labels=labels, training=True)
            loss = self.loss_fn(predictions, labels=labels)
            gradients = tape.gradient(loss, self.model.trainable_variables)  # TODO: Put gradient calculation in graph
            self.__optimize(gradients)

            return loss, gradients

    @tf.function
    def __optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def prediction(self, features, labels):
        return self.model(features, labels=labels, training=False)
