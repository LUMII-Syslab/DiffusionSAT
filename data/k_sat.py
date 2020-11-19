import random

import numpy as np
import tensorflow as tf
from pysat.formula import CNF
from pysat.solvers import Cadical

from data.dimac import DIMACDataset
from loss.sat import softplus_log_square_loss


class KSATVariables(DIMACDataset):

    def __init__(self, data_dir, force_data_gen=False, **kwargs) -> None:
        super(KSATVariables, self).__init__(data_dir, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 100000
        self.test_size = 5000
        self.min_vars = 3
        self.max_vars = 10

        self.p_k_2 = 0.3
        self.p_geo = 0.4

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:
        for _ in range(size):
            n_vars = random.randint(self.min_vars, self.max_vars)

            solver = Cadical()
            iclauses = []

            while True:
                k_base = 1 if random.random() < self.p_k_2 else 2
                k = k_base + np.random.geometric(self.p_geo)
                iclause = self.__generate_k_iclause(n_vars, k)

                solver.add_clause(iclause)
                is_sat = solver.solve()

                if is_sat:
                    iclauses.append(iclause)
                else:
                    break

            iclause_unsat = iclause
            iclause_sat = [-iclause_unsat[0]] + iclause_unsat[1:]

            iclauses.append(iclause_unsat)
            # yield only SAT instance
            # yield n_vars, self.prune(iclauses)

            iclauses[-1] = iclause_sat
            yield n_vars, self.remove_duplicate_clauses(iclauses)

    @staticmethod
    def __generate_k_iclause(n, k):
        vs = np.random.choice(n, size=min(n, k), replace=False)
        return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

    # TODO: remove subsumed clauses - when shorter clause is fully in a longer one, the longer one is redundant
    @staticmethod
    def remove_duplicate_clauses(clauses):
        return list({tuple(sorted(x)) for x in clauses})

    def create_adj_matrices(self, data):
        dense_shape = tf.stack([tf.reduce_sum(data["variable_count"]), tf.reduce_sum(data["clauses_in_formula"])])
        dense_shape = tf.cast(dense_shape, tf.int64)

        return {
            "adjacency_matrix_pos": self.create_adjacency_matrix(data["adj_indices_pos"], dense_shape),
            "adjacency_matrix_neg": self.create_adjacency_matrix(data["adj_indices_neg"], dense_shape),
            "clauses": tf.cast(data["batched_clauses"], tf.int32),
            "variable_count": data["variable_count"],
            "normal_clauses": data["clauses"]
        }

    @staticmethod
    def create_adjacency_matrix(indices, dense_shape):
        return tf.sparse.SparseTensor(indices,
                                      tf.ones(tf.shape(indices)[0], dtype=tf.float32),
                                      dense_shape=dense_shape)

    def prepare_dataset(self, dataset: tf.data.Dataset):
        return dataset.map(self.create_adj_matrices, tf.data.experimental.AUTOTUNE)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
                                  tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def loss(self, predictions, clauses):
        loss = 0.0
        for logits in predictions:  # TODO: Rewrite this without loop
            per_clause = softplus_log_square_loss(logits, clauses)
            loss += tf.reduce_sum(per_clause)

        step_count = tf.shape(predictions)[0]
        step_count = tf.cast(step_count, dtype=tf.float32)
        return loss / step_count

    def filter_loss_inputs(self, step_data) -> dict:
        return {"clauses": step_data["clauses"]}

    def filter_model_inputs(self, step_data) -> dict:  # TODO: Not good because dataset needs to know about model
        return {
            "adj_matrix_pos": step_data["adjacency_matrix_pos"],
            "adj_matrix_neg": step_data["adjacency_matrix_neg"],
            "clauses": step_data["clauses"]
        }

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32)],
                 experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
    def interpret_model_output(self, model_output):
        return tf.squeeze(model_output[-1], axis=-1)  # Take logits only from the last step

    @staticmethod
    def split_batch(predictions, variable_count):
        batched_logits = []  # TODO: Can I do it better?
        i = 0
        for length in variable_count:
            batched_logits.append(predictions[i:i + length])
            i += length

        return batched_logits

    def accuracy(self, prediction, step_data):
        prediction = tf.round(tf.sigmoid(prediction))
        prediction = self.split_batch(prediction, step_data["variable_count"])

        mean_acc = tf.metrics.Mean()
        mean_total_acc = tf.metrics.Mean()

        for pred, clause in zip(prediction, step_data["normal_clauses"]):
            accuracy, total_accuracy = self.__accuracy_for_single(pred, clause)
            mean_acc.update_state(accuracy)
            mean_total_acc.update_state(total_accuracy)

        return mean_acc.result(), mean_total_acc.result()

    @staticmethod
    def __accuracy_for_single(prediction, clauses):
        formula = CNF(from_clauses=[x.tolist() for x in clauses.numpy()])  # TODO: is there better way?
        with Cadical(bootstrap_with=formula.clauses) as solver:
            assum = [i if prediction[i - 1] == 1 else -i for i in range(1, len(prediction), 1)]
            correct_pred = solver.solve(assumptions=assum)
            solver.solve()
            variables = np.array(solver.get_model())
            variables[variables < 0] = 0
            variables[variables > 0] = 1

            equal_elem = np.equal(prediction, variables)
            correct = np.sum(equal_elem)
            total = prediction.shape[0]

        return correct / total, 1 if correct_pred else 0


class KSATLiterals(KSATVariables):

    def create_adj_matrices(self, data):
        var_count = tf.reduce_sum(data["variable_count"])
        neg = data["adj_indices_neg"]

        shape = [tf.shape(neg)[0], 1]

        offset = tf.ones(shape, dtype=tf.int32) * var_count
        zeros = tf.zeros(shape, dtype=tf.int32)
        offset = tf.concat([offset, zeros], axis=-1)
        offset = tf.cast(offset, tf.int64)
        neg = neg + offset

        dense_shape = tf.stack([var_count * 2, tf.reduce_sum(data["clauses_in_formula"])])
        dense_shape = tf.cast(dense_shape, tf.int64)

        adj = tf.concat([data["adj_indices_pos"], neg], axis=0)

        return {
            "adjacency_matrix": self.create_adjacency_matrix(adj, dense_shape),
            "clauses": tf.cast(data["batched_clauses"], tf.int32),
            "variable_count": data["variable_count"],
            "normal_clauses": data["clauses"]
        }

    def filter_loss_inputs(self, step_data) -> dict:
        return {"clauses": step_data["clauses"]}

    def filter_model_inputs(self, step_data) -> dict:
        return {
            "adj_matrix": step_data["adjacency_matrix"],
            "clauses": step_data["clauses"]
        }
