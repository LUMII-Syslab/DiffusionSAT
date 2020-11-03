from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense

import utils.shuffle as shuffle_utils


class MatrixSE(tf.keras.Model):

    def __init__(self, feature_maps=64):
        super(MatrixSE, self).__init__()
        self.input_layer = Dense(feature_maps, activation=tf.nn.relu)
        self.benes = BenesBlock(1, feature_maps)
        self.output_layer = Dense(1)

    # TODO: @tf.function
    def call(self, inputs, labels=None, training=None, mask=None):
        hidden = self.input_layer(inputs)
        hidden = self.benes(hidden, training=training)
        return self.output_layer(hidden)

    # TODO: Move this to main.py
    # def model(self, n, batchsize):
    #     x = Input(shape=(n, n, 1), batch_size=batchsize)
    #     return Model(inputs=[x], outputs=self.call(x))


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
    """Implementation of Quaternary Beneš block
    This implementation expects 4-D inputs - [batch_size, width, height, channels]
    Output shape will be same as input shape, expect channels will be in size of num_units.
    BenesBlock output is output from the last BenesBlock layer. No additional output processing is applied.
    """

    def __init__(self, block_count, num_units, **kwargs):
        """
        :param block_count: Determines Beneš block count that are chained together
        :param fixed_shuffle: Use fixed shuffle (equal in every layer) or dynamic (shuffle differs in every layer)
        :param num_units: Num units to use in Beneš block
        """
        super().__init__(**kwargs)
        self.block_count = block_count
        self.num_units = num_units
        self.block_layers = {}
        for i in range(self.block_count):
            self.block_layers[i] = {
                "forward": QuaternarySwitchUnit("forward", dropout_rate=0.1),
                "middle": QuaternarySwitchUnit("middle", dropout_rate=0.1),
                "reverse": QuaternarySwitchUnit("reverse", dropout_rate=0.1)
            }

        self.output_layer = Dense(self.num_units, name="output", )

    def call(self, inputs, training=False, **kwargs):
        input_shape = inputs.get_shape().as_list()
        level_count = (input_shape[1] - 1).bit_length() - 1

        last_layer = ZOrderFlatten()(inputs)

        for block_nr in tf.range(self.block_count):

            with tf.name_scope(f"benes_block_{block_nr}"):
                for _ in range(level_count):
                    switch = self.block_layers[block_nr]["forward"](last_layer, training=training)
                    last_layer = QuaternaryShuffleLayer(ShuffleType.RIGHT)(switch)

                for level in range(level_count):
                    last_layer = self.block_layers[block_nr]["reverse"](last_layer, training=training)
                    last_layer = QuaternaryShuffleLayer(ShuffleType.LEFT)(last_layer)

                last_layer = self.block_layers[block_nr]["middle"](last_layer, training=training)

        return ZOrderUnflatten()(last_layer, width=input_shape[1], height=input_shape[2])
