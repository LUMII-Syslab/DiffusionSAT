from pathlib import Path

import numpy as np
import tensorflow as tf
from pysat.formula import CNF
from pysat.solvers import Glucose4

from metrics.base import Metric


class SATAccuracyTF(Metric):

    def __init__(self) -> None:
        self.mean_acc = tf.metrics.Mean()
        self.mean_total_acc = tf.metrics.Mean()

    def update_state(self, model_output, step_data):
        acc, total_acc = self.__accuracy(model_output["prediction"], step_data)
        self.mean_acc.update_state(acc)
        self.mean_total_acc.update_state(total_acc)

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        mean_acc, mean_total_acc = self.__calc_accuracy(reset_state)

        with tf.name_scope("accuracy"):
            tf.summary.scalar("accuracy", mean_acc, step=step)
            tf.summary.scalar("total_accuracy", mean_total_acc, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        mean_acc, mean_total_acc = self.__calc_accuracy(reset_state)
        print(f"Accuracy: {mean_acc.numpy():.4f}")
        print(f"Total fully correct: {mean_total_acc.numpy():.4f}")

    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        mean_acc, mean_total_acc = self.__calc_accuracy(reset_state)
        lines = [prepend_str + '\n'] if prepend_str else []
        lines.append(f"Total fully correct: {mean_total_acc.numpy():.4f}\n")

        file_path = Path(file)
        with file_path.open("a") as file:
            file.writelines(lines)

    def reset_state(self):
        self.mean_acc.reset_states()
        self.mean_total_acc.reset_states()

    def __calc_accuracy(self, reset_state):
        mean_acc = self.mean_acc.result()
        mean_total_acc = self.mean_total_acc.result()

        if reset_state:
            self.reset_state()

        return mean_acc, mean_total_acc

    def __accuracy(self, predictions, step_data):
        predictions = tf.round(tf.sigmoid(predictions))
        equal_variables = tf.equal(tf.cast(predictions, tf.int32), step_data["solutions"].flat_values)
        equal_variables = tf.cast(equal_variables, tf.float32)
        correct = tf.reduce_sum(equal_variables)
        acc = correct / tf.cast(tf.reduce_sum(step_data["variable_count"]), tf.float32)

        clauses = step_data["clauses"]
        total_acc = self.is_sat_assignment(predictions, clauses, step_data["clauses_count"])

        return acc, total_acc

    @staticmethod
    def is_sat_assignment(predictions: tf.Tensor, clauses: tf.RaggedTensor, clauses_in_graph: tf.Tensor):
        clauses_split = clauses.row_lengths()
        flat_clauses = clauses.flat_values
        clauses_mask = tf.repeat(tf.range(0, clauses.nrows()), clauses_split)
        clauses_index = tf.abs(flat_clauses) - 1  # Just star indexing from 0. DIMACS standard start variables from 1
        vars_in_clause = tf.gather(predictions, clauses_index)  # Gather clauses of variables
        clauses_sat = tf.math.segment_max(vars_in_clause, clauses_mask)

        graph_count = tf.shape(clauses_in_graph)
        graph_id = tf.range(0, graph_count[0])
        graph_mask = tf.repeat(graph_id, clauses_in_graph)
        formula_sat = tf.math.segment_min(clauses_sat, graph_mask)

        return formula_sat


class SATAccuracy(Metric):

    def __init__(self) -> None:
        self.mean_acc = tf.metrics.Mean()
        self.mean_total_acc = tf.metrics.Mean()

    def update_state(self, model_output, step_data):
        acc, total_acc = self.__accuracy(model_output["prediction"], step_data)
        self.mean_acc.update_state(acc)
        self.mean_total_acc.update_state(total_acc)

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        mean_acc, mean_total_acc = self.__calc_accuracy(reset_state)

        with tf.name_scope("accuracy"):
            tf.summary.scalar("accuracy", mean_acc, step=step)
            tf.summary.scalar("total_accuracy", mean_total_acc, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        mean_acc, mean_total_acc = self.__calc_accuracy(reset_state)
        print(f"Accuracy: {mean_acc.numpy():.4f}")
        print(f"Total fully correct: {mean_total_acc.numpy():.4f}")

    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        mean_acc, mean_total_acc = self.__calc_accuracy(reset_state)
        lines = [prepend_str + '\n'] if prepend_str else []
        lines.append(f"Total fully correct: {mean_total_acc.numpy():.4f}\n")

        file_path = Path(file)
        with file_path.open("a") as file:
            file.writelines(lines)

    def reset_state(self):
        self.mean_acc.reset_states()
        self.mean_total_acc.reset_states()

    def __calc_accuracy(self, reset_state):
        mean_acc = self.mean_acc.result()
        mean_total_acc = self.mean_total_acc.result()

        if reset_state:
            self.reset_state()

        return mean_acc, mean_total_acc

    def __accuracy(self, prediction, step_data):
        prediction = tf.round(tf.sigmoid(prediction))
        prediction = self.__split_batch(prediction, step_data["variable_count"])

        clauses = step_data["normal_clauses"]
        acc = [self.__accuracy_for_single(pred, clause) for pred, clause in zip(prediction, clauses)]
        acc, total_acc = zip(*acc)

        return acc, total_acc

    @staticmethod
    def __split_batch(predictions, variable_count):
        batched_logits = []
        i = 0
        for length in variable_count:
            batched_logits.append(predictions[i:i + length])
            i += length

        return batched_logits

    @staticmethod
    def __accuracy_for_single(prediction, clauses):
        formula = CNF(from_clauses=[x.tolist() for x in clauses.numpy()])
        with Glucose4(bootstrap_with=formula.clauses) as solver:
            assum = [i if prediction[i - 1] == 1 else -i for i in range(1, len(prediction), 1)]
            correct_pred = solver.solve(assumptions=assum)
            is_sat = solver.solve()
            assert is_sat  # all instances in dataset should be satisfiable
            variables = np.array(solver.get_model())
            variables[variables < 0] = 0
            variables[variables > 0] = 1

            equal_elem = np.equal(prediction, variables)
            correct = np.sum(equal_elem)
            total = prediction.shape[0]

        return correct / total, 1 if correct_pred else 0
