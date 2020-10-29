import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense


class MLP(Model):
    def __init__(self, layer_count, hidden_nmap, out_nmap, activation=tf.nn.relu, out_activation=None, **kwargs):
        super().__init__(**kwargs)

        self.dense_layers = [Dense(hidden_nmap, activation=activation) for _ in range(layer_count - 1)]
        self.dense_layers.append(Dense(out_nmap, activation=out_activation))

    def call(self, inputs, training=None, mask=None):
        current = inputs
        for layer in self.dense_layers:
            current = layer(current, training=training)
        return current
