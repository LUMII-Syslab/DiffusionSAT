import tensorflow as tf
from model.mlp import MLP


class DenseGNN(tf.keras.layers.Layer):
    """
    Graph Neural Network for a full graph, given as an adjacency matrix
    """

    def __init__(self,  **kwargs):
        super(DenseGNN, self).__init__(**kwargs)


    def build(self, input_shape):
        feature_maps = input_shape[-1]
        hidden_maps = feature_maps * 2
        self.incoming_edge_mlp = MLP(3, hidden_maps, feature_maps, name="incoming_edge_mlp", do_layer_norm=True, norm_axis = [1,2])
        self.outgoing_edge_mlp = MLP(3, hidden_maps, feature_maps, name="outgoing_edge_mlp", do_layer_norm=True, norm_axis = [1,2])
        #self.vertex_mlp = MLP(3, hidden_maps, feature_maps, name="vertex_mlp", do_layer_norm=True)
        self.edge_mlp = MLP(3, hidden_maps*2, feature_maps, name="edge_mlp", do_layer_norm=True, norm_axis = [1,2])
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
        sqrt_n = tf.sqrt(tf.cast(n_vertices, tf.float32))# todo: does it generalize to other size graphs?

        # calculate data for each incoming and outgoing vertex for each edge
        incoming_state = self.incoming_edge_mlp(inputs, training = training)
        outgoing_state = self.outgoing_edge_mlp(inputs, training = training)

        # vertex state is formed as sum over its incoming and outgoing edge data
        incoming_state = tf.reduce_sum(incoming_state, axis=1) / sqrt_n
        outgoing_state = tf.reduce_sum(outgoing_state, axis=2) / sqrt_n
        vertex_state = tf.concat([incoming_state, outgoing_state], axis=-1)
        #vertex_state = self.vertex_mlp(vertex_state)

        # the new value for each edge is calculated from its in- and out-vertex data and the previous edge state
        vertex_tile_in = tf.tile(tf.expand_dims(vertex_state, 1), [1, n_vertices, 1, 1])
        vertex_tile_out = tf.tile(tf.expand_dims(vertex_state, 2), [1, 1, n_vertices, 1])
        edge_unit = tf.concat([inputs, vertex_tile_in, vertex_tile_out], axis = -1)

        output = self.edge_mlp(edge_unit)
        return output
