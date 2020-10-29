import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from pysat.formula import CNF
from pysat.solvers import Cadical

from data.dataset import Dataset
from loss.sat import variables_mul_loss
from mk_problem import Problem


class NeuroSATDataset(Dataset):

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
        data = data.map(lambda x, y, dense_shape: (
            tf.sparse.SparseTensor(x, tf.ones(tf.cast(tf.shape(x)[0], tf.int32), dtype=tf.float32),
                                   dense_shape=dense_shape),
            tf.cast(y, tf.int32)
        ))
        data = data.shuffle(10000)
        data = data.repeat()

        return data

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
        normal_clauses = tf.ragged.constant(normal_clauses, dtype=tf.int32,
                                            row_splits_dtype=tf.int32)  # TODO: this is same as clauses, remove it

        data_add = tf.data.Dataset.from_tensor_slices((var_count, normal_clauses))
        data = tf.data.Dataset.from_tensor_slices((indices, clauses, n_lits, n_clauses))
        data = data.map(
            lambda x, y, z, v: (
                tf.cast(x.to_tensor(), tf.int64), y, tf.cast(tf.stack([z, v]), tf.int64)))
        data = data.map(lambda x, y, dense_shape: (
            tf.sparse.SparseTensor(x, tf.ones(tf.cast(tf.shape(x)[0], tf.int32), dtype=tf.float32),
                                   dense_shape=dense_shape),
            tf.cast(y, tf.int32)
        ))

        return tf.data.Dataset.zip((data, data_add))

    def loss_fn(self, predictions, labels=None):
        predictions = tf.expand_dims(predictions, axis=-1)
        loss = variables_mul_loss(predictions, labels)
        return tf.reduce_mean(loss)

    def accuracy_fn(self, prediction, label=None):
        formula = CNF(from_clauses=[x.tolist() for x in label.numpy()])  # TODO: is there better way?
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

            if correct_pred:
                return 1

        return correct / total
