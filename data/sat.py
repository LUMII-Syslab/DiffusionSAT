import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from pysat.formula import CNF
from pysat.solvers import Cadical

from data.dataset import Dataset
from loss.sat import softplus_log_square_loss
from mk_problem import Problem


class RandomKSAT(Dataset):

    def __init__(self) -> None:
        train_dir = 'data_files/train/sr5'
        test_dir = 'data_files/test/sr5'

        self.train_problems = self.load_problems(train_dir)
        self.test_problems = self.load_problems(test_dir)

    @staticmethod
    def load_problems(directory):
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory \"{str(dir_path)}\" doesn't exist!")

        problems = []
        files = [file for file in dir_path.iterdir() if file.is_file()]

        for file in files:
            with file.open('rb') as f:
                problems += pickle.load(f)

        return problems

    @staticmethod
    def create_example(problem: Problem):
        return problem.L_unpack_indices, problem.clauses, problem.n_lits, problem.n_clauses

    def train_data(self) -> tf.data.Dataset:
        # TODO: remove constants (remove pickle and read from files)
        indices, clauses, n_lits, n_clauses = list(
            zip(*[self.create_example(problem) for problem in self.train_problems]))
        indices = tf.ragged.constant(indices, dtype=tf.int32, row_splits_dtype=tf.int32)
        clauses = tf.ragged.constant(clauses, dtype=tf.int32, row_splits_dtype=tf.int32)
        n_lits = tf.constant(n_lits)
        n_clauses = tf.constant(n_clauses)

        data = tf.data.Dataset.from_tensor_slices((indices, clauses, n_lits, n_clauses))
        data = data.map(lambda x, y, z, v: (tf.cast(x.to_tensor(), tf.int64), y, tf.cast(tf.stack([z, v]), tf.int64)))
        data = data.map(lambda x, y, dense_shape: {"adjacency_matrix": self.create_adjacency_matrix(x, dense_shape),
                                                   "clauses": tf.cast(y, tf.int32)})
        data = data.shuffle(10000)
        data = data.repeat()  # TODO: Move shuffling, repetition, batching to common code

        return data

    @staticmethod
    def create_adjacency_matrix(indices, dense_shape):
        return tf.sparse.SparseTensor(indices,
                                      tf.ones(tf.cast(tf.shape(indices)[0], tf.int32), dtype=tf.float32),
                                      dense_shape=dense_shape)

    def validation_data(self) -> tf.data.Dataset:
        data = self.test_data()
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def test_data(self) -> tf.data.Dataset:
        # TODO: Clear up this garbage
        test_data = [(*self.create_example(p), p.variable_count, p.normal_clauses) for p in self.test_problems]
        indices, clauses, n_lits, n_clauses, var_count, normal_clauses = list(zip(*test_data))
        indices = tf.ragged.constant(indices, dtype=tf.int32, row_splits_dtype=tf.int32)
        clauses = tf.ragged.constant(clauses, dtype=tf.int32, row_splits_dtype=tf.int32)
        n_lits = tf.constant(n_lits)
        n_clauses = tf.constant(n_clauses)
        var_count = tf.ragged.constant(var_count, dtype=tf.int32, row_splits_dtype=tf.int32)
        normal_clauses = tf.ragged.constant(normal_clauses, dtype=tf.int32, row_splits_dtype=tf.int32)

        data_add = tf.data.Dataset.from_tensor_slices((var_count, normal_clauses))
        data = tf.data.Dataset.from_tensor_slices((indices, clauses, n_lits, n_clauses))
        data = data.map(lambda x, y, z, v: (tf.cast(x.to_tensor(), tf.int64), y, tf.cast(tf.stack([z, v]), tf.int64)))
        data = data.map(lambda x, y, dense_shape: (self.create_adjacency_matrix(x, dense_shape), tf.cast(y, tf.int32)))

        data = tf.data.Dataset.zip((data, data_add))
        data = data.map(lambda x, y: {"adjacency_matrix": x[0],
                                      "clauses": x[1],
                                      "variable_count": y[0],
                                      "normal_clauses": y[1]
                                      })
        return data

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
        return {"adj_matrix": step_data["adjacency_matrix"], "clauses": step_data["clauses"]}

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
