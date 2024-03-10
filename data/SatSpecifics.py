import tensorflow as tf

from data.dimac import TaskSpecifics
from config import Config
from metrics.sat_metrics import SATAccuracyTF, StepStatistics

class SatSpecifics(TaskSpecifics):
    
    def __init__(self, input_mode=Config.input_mode, **kwargs):
        self.args_filter = self.__prepare_filter(input_mode)
    
    def prepare_dataset(self, dataset: tf.data.Dataset):
        return dataset.map(self.create_adj_matrices, tf.data.experimental.AUTOTUNE)

    def args_for_train_step(self, step_data) -> dict:
        return self.args_filter(step_data)

    def metrics(self, initial=False) -> list:
        return [SATAccuracyTF(), StepStatistics()]
    
    def create_adj_matrices(self, data):
        var_count = tf.reduce_sum(data["variable_count"])
        clauses_count = tf.reduce_sum(data["clauses_in_formula"])

        shape = [tf.shape(data["adj_indices_neg"])[0], 1]
        offset = tf.ones(shape, dtype=tf.int32) * var_count  # offset for the negative variable nodes (=== +var_count for each negative node)
        zeros = tf.zeros(shape, dtype=tf.int32)              # offset for the clause nodes (zeros)
        offset = tf.concat([offset, zeros], axis=-1)
        offset = tf.cast(offset, tf.int64)
        neg = data["adj_indices_neg"] + offset

        lit_shape = self.create_shape(var_count * 2, clauses_count)
        var_shape = self.create_shape(var_count, clauses_count)
        adj_matrix_lit = tf.concat([data["adj_indices_pos"], neg], axis=0)
        adj_matrix_lit = self.create_adjacency_matrix(adj_matrix_lit, lit_shape)

        adj_matrix_pos = self.create_adjacency_matrix(data["adj_indices_pos"], var_shape)
        adj_matrix_neg = self.create_adjacency_matrix(data["adj_indices_neg"], var_shape)

        graph_count = tf.shape(data["variable_count"])
        graph_id = tf.range(0, graph_count[0])
        variables_mask = tf.repeat(graph_id, data["variable_count"])   # to which graph each of the combined (batched) variables belongs
        clauses_mask = tf.repeat(graph_id, data["clauses_in_formula"]) # to which graph each of the combined (batched) clauses belongs

        clauses_enum = tf.range(0, var_shape[1], dtype=tf.int32)
        c_g_indices = tf.stack([clauses_mask, clauses_enum], axis=1)
        c_g_indices = tf.cast(c_g_indices, tf.int64)  # list of pairs: [graph_id, clause#]
        clauses_graph_adj = self.create_adjacency_matrix(c_g_indices,
                                                         self.create_shape(tf.cast(graph_count[0], tf.int64),
                                                                           var_shape[1]))

        variables_enum = tf.range(0, var_shape[0], dtype=tf.int32)
        v_g_indices = tf.stack([variables_mask, variables_enum], axis=1)
        v_g_indices = tf.cast(v_g_indices, tf.int64)  # list of pairs: [graph_id, var#]
        variables_graph_adj = self.create_adjacency_matrix(v_g_indices,
                                                           self.create_shape(tf.cast(graph_count[0], tf.int64),
                                                                             var_shape[0]))

        return {
            "adjacency_matrix_pos": adj_matrix_pos,
            "adjacency_matrix_neg": adj_matrix_neg,
            "adjacency_matrix": adj_matrix_lit,
            "clauses": tf.cast(data["batched_clauses"], tf.int32),
            "variables_in_graph": data["variable_count"],
            "normal_clauses": data["clauses"],
            "clauses_graph_adj": clauses_graph_adj,
            "variables_graph_adj": variables_graph_adj,
            "solutions": data["solutions"]
        }

    @staticmethod
    def create_shape(variables, clauses):
        dense_shape = tf.stack([variables, clauses])
        return tf.cast(dense_shape, tf.int64)

    @staticmethod
    def create_adjacency_matrix(indices, dense_shape):
        return tf.sparse.SparseTensor(indices,
                                      tf.ones(tf.shape(indices)[0], dtype=tf.float32),
                                      dense_shape=dense_shape)

    def __prepare_filter(self, input_mode):
        # returns a function that takes step_data (batched), extracts some attributes, and renames them
        # for passing to model.train_step(...<here>...)
        if input_mode == "variables":
            return lambda step_data: {
                "adj_matrix_pos": step_data["adjacency_matrix_pos"],
                "adj_matrix_neg": step_data["adjacency_matrix_neg"],
                "clauses_graph": step_data["clauses_graph_adj"],
                "variables_graph": step_data["variables_graph_adj"],
                "solutions": step_data["solutions"]
            }
        elif input_mode == "literals":
            return lambda step_data: {
                "adj_matrix": step_data["adjacency_matrix"],
                "clauses_graph": step_data["clauses_graph_adj"],
                "variables_graph": step_data["variables_graph_adj"],
                "solutions": step_data["solutions"]
            }
        else:
            raise NotImplementedError(
                f"{input_mode} is not registered. Available modes \"literals\" or \"variables\"")


