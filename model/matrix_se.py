from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense

import utils.shuffle as shuffle_utils


class MatrixSE(tf.keras.Model):

    def __init__(self, feature_maps=64, output_features=1, block_count=1, **kwargs):
        super(MatrixSE, self).__init__(**kwargs)
        self.input_layer = Dense(feature_maps, activation=tf.nn.relu, name="input_layer")
        self.benes_blocks = [BenesBlock() for _ in range(block_count)]
        self.output_layer = Dense(output_features, name="output_layer")

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: tensor [batch_size, height, width, feature_maps], where height == width
        :param training: boolean
        :param mask: not used
        :return: logit tensor in shape [batch_size, height, width, output_features]
        """
        input_shape = inputs.get_shape().as_list()

        inputs = tf.expand_dims(inputs, axis=-1)
        hidden = self.input_layer(inputs)

        hidden = ZOrderFlatten()(hidden)
        for block in self.benes_blocks:
            hidden = block(hidden, training=training)
        hidden = ZOrderUnflatten()(hidden, width=input_shape[1], height=input_shape[2])

        return self.output_layer(hidden)


class QuaternarySwitchUnit(tf.keras.layers.Layer):

    def __init__(self, name, channel_count=4, dropout_rate=0.1, **kwargs):
        super(QuaternarySwitchUnit, self).__init__(name=name, **kwargs)
        self.channel_count = channel_count
        self.dropout_rate = dropout_rate
        self.residual_weight = 0.9
        self.candidate_weight = np.sqrt(1 - self.residual_weight ** 2) * 0.25
        self.scale_init = np.log(self.residual_weight / (1 - self.residual_weight))

        self.num_units = None
        self.reshaped_units = None
        self.residual_scale = None
        self.layer_norm = None
        self.dropout = None

        self.linear_one = None
        self.linear_two = None

    def build(self, input_shape):
        self.num_units = input_shape.as_list()[2]
        self.reshaped_units = self.num_units * self.channel_count

        initializer = tf.constant_initializer(self.scale_init)
        self.residual_scale = self.add_weight("residual", [self.reshaped_units], initializer=initializer)

        self.linear_one = Dense(self.reshaped_units * 2, name="linear_one", use_bias=False)
        self.linear_two = Dense(self.reshaped_units, name="linear_two")

        self.layer_norm = LayerNormalization(subtract_mean=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=False, **kwargs):
        batch_size, length, num_units = inputs.shape.as_list()[:3]
        inputs = tf.reshape(inputs, shape=[batch_size, length // self.channel_count, self.reshaped_units])
        dropout = self.dropout(inputs, training=training)

        first_linear = self.linear_one(dropout)
        norm = self.layer_norm(first_linear)
        gelu = tfa.activations.gelu(norm)  # TODO: In next tensorflow version replace with verison from core
        second_linear = self.linear_two(gelu)

        residual_scale = tf.nn.sigmoid(self.residual_scale)
        candidate = residual_scale * inputs + second_linear * self.candidate_weight
        return tf.reshape(candidate, [batch_size, length, self.num_units])


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self, axis=1, epsilon=1e-6, subtract_mean=True, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        self.bias = None
        self.subtract_mean = subtract_mean
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        num_units = input_shape.as_list()[-1]
        self.bias = self.add_weight("bias", [1, 1, num_units], initializer=tf.zeros_initializer)

    def call(self, inputs, **kwargs):
        if self.subtract_mean:  # subtracting mean may not be necessary: https://arxiv.org/abs/1910.07467
            inputs -= tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
            inputs += self.bias

        variance = tf.reduce_mean(tf.square(inputs), self.axis, keepdims=True)
        return inputs * tf.math.rsqrt(variance + self.epsilon)


class ShuffleType(Enum):
    LEFT = shuffle_utils.qrol
    RIGHT = shuffle_utils.qror

    def __call__(self, *args, **kwargs):
        self.value(*args)


class QuaternaryShuffleLayer(tf.keras.layers.Layer):
    """ Implements left quaternary cyclic shift for input tensor as described in
     "Two-Dimensional Benes Network" by Uh-Sock Rhee and Mir M. Mirsalehi
    """

    def __init__(self, shuffle_type: ShuffleType, layer_level=0, **kwargs):
        super(QuaternaryShuffleLayer, self).__init__(trainable=False, **kwargs)
        self.level = layer_level
        self.shuffled_indices = None
        self.shuffle = shuffle_type

    def build(self, input_shape: tf.TensorShape):
        _, length, _ = input_shape.as_list()
        digits = shuffle_utils.quaternary_digits(length - 1)
        self.shuffled_indices = [self.shuffle(x, digits, self.level) for x in range(length)]

    def call(self, inputs, **kwargs):
        return tf.gather(inputs, self.shuffled_indices, axis=1)


class ZOrderFlatten(tf.keras.layers.Layer):
    """ Implements flattening by quaternary indices as described
    in "Two-Dimensional Benes Network" by Uh-Sock Rhee and Mir M. Mirsalehi
    """

    def call(self, inputs, **kwargs):
        batch_size, width, height, *channels = inputs.shape.as_list()
        vec_size = width * height

        matrix = np.reshape(np.arange(vec_size), [width, height]).tolist()
        quaternary_mask = shuffle_utils.matrix_to_vector(matrix)

        inputs = tf.reshape(inputs, [batch_size, vec_size] + channels)
        return tf.gather(inputs, quaternary_mask, axis=1)


class ZOrderUnflatten(tf.keras.layers.Layer):
    """ Implements vector reshaping to matrix by quaternary indices as described
    in "Two-Dimensional Benes Network" by Uh-Sock Rhee and Mir M. Mirsalehi
    """

    def call(self, inputs, width=None, height=None, **kwargs):
        _, length, nmaps = inputs.shape.as_list()

        matrix = shuffle_utils.vector_to_matrix([x for x in range(length)])
        quaternary_mask = np.reshape(np.array(matrix), [length])

        gather = tf.gather(inputs, quaternary_mask, axis=1)
        return tf.reshape(gather, [-1, width, height, nmaps])


class BenesBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.forward = QuaternarySwitchUnit("forward", dropout_rate=0.1)
        self.reverse = QuaternarySwitchUnit("reverse", dropout_rate=0.1)
        self.middle = QuaternarySwitchUnit("middle", dropout_rate=0.1)
        self.shuffle_forward = QuaternaryShuffleLayer(ShuffleType.RIGHT)
        self.shuffle_reverse = QuaternaryShuffleLayer(ShuffleType.LEFT)

    def call(self, inputs, training=False, **kwargs):
        side = tf.cast(tf.shape(inputs)[1], dtype=tf.float32)
        bit_length = tf.math.log(side - 1) / tf.math.log(2.)
        level_count = tf.cast(tf.math.floor(bit_length), tf.int32)

        last_layer = inputs

        for _ in tf.range(level_count):
            last_layer = self.forward(last_layer, training=training)
            last_layer = self.shuffle_forward(last_layer)

        for _ in tf.range(level_count):
            last_layer = self.reverse(last_layer, training=training)
            last_layer = self.shuffle_reverse(last_layer)

        return self.middle(last_layer, training=training)
