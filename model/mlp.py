import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda

from layers.layer_normalization import LayerNormalization

def leaky_gelu(x):
    sg = tf.sigmoid(1.702*x)
    return x * (0.8 * sg + 0.2)


class MLP(Model):
    def __init__(self, layer_count, hidden_nmap,
                 out_nmap, activation=tf.nn.leaky_relu,
                 out_activation = None, out_bias = None,
                 do_layer_norm = False, norm_axis = 0, **kwargs):

        super().__init__(**kwargs)

        if not do_layer_norm:
            self.dense_layers = [Dense(hidden_nmap, activation=activation) for _ in range(layer_count - 1)]
        else:
            self.dense_layers = []
            for i in range(layer_count - 1):
                self.dense_layers.append(Dense(hidden_nmap, activation=None, use_bias=i > 0))
                if i == 0: self.dense_layers.append(LayerNormalization(axis=norm_axis, subtract_mean=True))
                if activation is not None: self.dense_layers.append(Lambda(lambda x: activation(x)))

        #self.dense_layers.append(tf.keras.layers.Dropout(rate=0.05))
        bias = 'zeros'
        if out_bias is not None: bias = tf.constant_initializer(out_bias)
        self.dense_layers.append(Dense(out_nmap, activation=out_activation, bias_initializer=bias))

    def call(self, inputs, training=None, mask=None):
        current = inputs
        for layer in self.dense_layers:
            current = layer(current, training=training)
        return current
