import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense, Lambda

from layers.normalization import LayerNormalization


class MLP(Model):
    def __init__(self, layer_count, hidden_nmap,
                 out_nmap, activation=tf.nn.leaky_relu,
                 out_activation=None, out_bias=None,
                 do_layer_norm=False, norm_axis=0,
                 normalizer=None,
                 dropout_rate=0.0,
                 init_zero=False, **kwargs):

        super().__init__(**kwargs)

        if not do_layer_norm:
            self.dense_layers = [Dense(hidden_nmap, activation=activation) for _ in range(layer_count - 1)]
        else:
            self.dense_layers = []
            if normalizer is None: normalizer = LayerNormalization(axis=norm_axis, subtract_mean=True)
            for i in range(layer_count - 1):
                self.dense_layers.append(Dense(hidden_nmap, activation=None, use_bias=i > 0))
                if i == 0: self.dense_layers.append(normalizer)
                if activation is not None: self.dense_layers.append(Lambda(lambda x: activation(x)))

        if dropout_rate > 0: self.dense_layers.append(tf.keras.layers.Dropout(rate=dropout_rate))
        bias = 'zeros'
        if out_bias is not None: bias = tf.constant_initializer(out_bias)
        if init_zero:
            self.dense_layers.append(
                Dense(out_nmap, activation=out_activation, kernel_initializer='zeros', bias_initializer=bias))
        else:
            self.dense_layers.append(Dense(out_nmap, activation=out_activation, bias_initializer=bias))
        self.normalizer = normalizer

    @tf.function
    def call(self, inputs, training=None):
        current = inputs
        for layer in self.dense_layers:
            if layer == self.normalizer:
                current = layer(current, training=training)
            else:
                current = layer(current, training=training)
        return current


class NeuroCoreMLP(Model):
    def __init__(self, layer_count, hidden_nmap, out_nmap, activation=tf.nn.relu6, **kwargs):
        super().__init__(**kwargs)
        self.dense_layers = [Dense(hidden_nmap, kernel_initializer='glorot_normal', activation=activation) for _ in
                             range(layer_count - 1)]
        self.dense_layers.append(Dense(out_nmap, kernel_initializer='glorot_normal'))

    @tf.function
    def call(self, inputs, training=None):
        current = inputs
        for layer in self.dense_layers:
            current = layer(current, training=training)

        return current
