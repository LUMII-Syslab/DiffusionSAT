import numpy as np
import tensorflow as tf
from model.mlp import MLP

def inv_sigmoid(y):
    return np.log(y / (1 - y))


class DenseGNN(tf.keras.layers.Layer):
    """
    Graph Neural Network for a full graph, given as an adjacency matrix
    """

    def __init__(self,  **kwargs):
        super(DenseGNN, self).__init__(**kwargs)
        self.residual_weight_initial_value = 0.5


    def build(self, input_shape):
        feature_maps = input_shape[-1]
        hidden_maps = feature_maps * 2
        self.incoming_edge_mlp = MLP(3, hidden_maps, feature_maps, name="incoming_edge_mlp", do_layer_norm=True, norm_axis = [1,2])
        self.outgoing_edge_mlp = MLP(3, hidden_maps, feature_maps, name="outgoing_edge_mlp", do_layer_norm=True, norm_axis = [1,2])
        self.edge_mlp = MLP(3, hidden_maps*2, feature_maps, name="edge_mlp", do_layer_norm=True, norm_axis = [1,2])

        self.prev_weight = self.add_weight("prev_weight", shape = [feature_maps],
                                           initializer=tf.constant_initializer(inv_sigmoid(self.residual_weight_initial_value)))
        self.candidate_weight = self.add_weight("cand_weight", shape=[feature_maps], initializer = tf.zeros_initializer())

        return super().build(input_shape)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        :param inputs: tensor [batch_size, height, width, feature_maps], where height == width
        :param training: boolean
        :param mask: not used
        :return: logit tensor in shape [batch_size, height, width, feature_maps]
        """
        input_shape = inputs.get_shape().as_list()
        n_vertices = input_shape[1]
        sqrt_n = tf.sqrt(tf.cast(n_vertices, tf.float32))  # todo: does it generalize to other size graphs?

        feature_maps = input_shape[3]
        mask = tf.expand_dims(mask, axis=3)
        mask = tf.repeat(mask, repeats=feature_maps, axis=3)

        # calculate data for each incoming and outgoing vertex for each edge
        incoming_state = self.incoming_edge_mlp(inputs * mask, training=training)
        outgoing_state = self.outgoing_edge_mlp(inputs * mask, training=training)

        # vertex state is formed as sum over its incoming and outgoing edge data
        incoming_state = tf.reduce_sum(incoming_state * mask, axis=1) / sqrt_n
        outgoing_state = tf.reduce_sum(outgoing_state * mask, axis=2) / sqrt_n
        vertex_state = tf.concat([incoming_state, outgoing_state], axis=-1)

        # the new value for each edge is calculated from its in- and out-vertex data and the previous edge state
        vertex_tile_in = tf.tile(tf.expand_dims(vertex_state, 1), [1, n_vertices, 1, 1])
        vertex_tile_out = tf.tile(tf.expand_dims(vertex_state, 2), [1, 1, n_vertices, 1])
        edge_unit = tf.concat([inputs, vertex_tile_in, vertex_tile_out], axis=-1)
        candidate = self.edge_mlp(edge_unit)

        # use ReZero residual connection
        final = inputs * tf.sigmoid(self.prev_weight) + candidate * self.candidate_weight

        return final
