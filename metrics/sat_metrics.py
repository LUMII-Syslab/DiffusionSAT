from pathlib import Path

import numpy as np
import tensorflow as tf
from pysat.formula import CNF
from pysat.solvers import Cadical

from metrics.base import Metric


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

        return np.mean(acc), np.mean(total_acc)

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
