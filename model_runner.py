import tensorflow as tf

from data.dataset import Dataset


class ModelRunner:

    def __init__(self,
                 model_fn: tf.keras.Model,
                 dataset: Dataset,
                 optimizer: tf.optimizers.Optimizer) -> None:
        self.model = model_fn
        self.dataset = dataset
        self.optimizer = optimizer

    def train_step(self, step_data):
        with tf.GradientTape() as tape:
            model_inputs = self.dataset.filter_model_inputs(step_data)
            predictions = self.model(**model_inputs, training=True)
            loss = self.dataset.loss(predictions, step_data)
            gradients = tape.gradient(loss, self.model.trainable_variables)  # TODO: Put gradient calculation in graph
            self.__optimize(gradients)

            return loss, gradients

    @tf.function
    def __optimize(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def prediction(self, step_data):
        model_inputs = self.dataset.filter_model_inputs(step_data)
        return self.model(**model_inputs, training=False)

    def print_summary(self, step_data):
        model_inputs = self.dataset.filter_model_inputs(step_data)
        self.model(**model_inputs, training=True)
        self.model.summary()
