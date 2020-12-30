from tensorflow.keras.layers import Layer
import tensorflow as tf


class EdgeDropout(Layer):

    def __init__(self, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = dropout_rate

    def call(self, inputs: tf.SparseTensor, training=False, **kwargs):
        values = inputs.values

        if training:
            values = tf.nn.dropout(values, self.rate)

        return tf.sparse.SparseTensor(inputs.indices, values, inputs.dense_shape)
