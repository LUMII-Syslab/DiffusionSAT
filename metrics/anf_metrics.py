from pathlib import Path
import tensorflow as tf
from metrics.base import Metric

class ANFAccuracyTF(Metric):

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

    def get_values(self, reset_state=True):
        return self.__calc_accuracy(reset_state)

    def __calc_accuracy(self, reset_state):
        mean_acc = self.mean_acc.result()
        mean_total_acc = self.mean_total_acc.result()

        if reset_state:
            self.reset_state()

        return mean_acc, mean_total_acc

    def __accuracy(self, predictions, step_data):
        predictions = tf.round(tf.sigmoid(predictions))
        predictions = tf.cast(predictions, tf.int32)
        solutions = step_data[5]
        variables_graph =  step_data[7]

        equal_variables = tf.equal(predictions, solutions)
        equal_variables = tf.cast(equal_variables, tf.float32)
        error = 1-equal_variables
        acc = tf.reduce_mean(equal_variables)
        per_graph_error = tf.minimum(tf.sparse.sparse_dense_matmul(variables_graph, tf.expand_dims(error,axis=-1)), 1.0)
        total_acc = 1-tf.reduce_mean(per_graph_error)

        return acc, total_acc
