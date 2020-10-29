import tensorflow as tf


class ModelRunner:

    def __init__(self,
                 model_fn: tf.keras.Model,
                 loss_fn: callable,
                 optimizer: tf.optimizers.Optimizer) -> None:
        self.model = model_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    @tf.function(  # TODO: Make this more general
        input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                         tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
        experimental_relax_shapes=True)
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(features, labels=labels, training=True)
            loss = self.loss_fn(predictions, labels=labels)
            train_vars = self.model.trainable_variables
            gradients = tape.gradient(loss, train_vars)
            self.optimizer.apply_gradients(zip(gradients, train_vars))
            return loss, gradients

    @tf.function(  # TODO: Make this more general
        input_signature=[tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32),
                         tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
        experimental_relax_shapes=True)
    def prediction(self, features, labels):
        return self.model(features, labels=labels, training=False)
