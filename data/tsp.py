import math

import numpy as np
import tensorflow as tf

from data.dataset import Dataset
from loss.tsp import tsp_loss


class EuclideanTSP(Dataset):

    # TODO(@Emīls): Move batch_size to config and add kwargs to datasets
    def __init__(self, n_vertices=8, dataset_size=1500, batch_size=15, **kwargs) -> None:
        self.n_vertices = n_vertices
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def train_data(self) -> tf.data.Dataset:
        data = self.__generate_data()
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def __generate_data(self) -> tf.data.Dataset:
        coords = np.random.rand(self.dataset_size, self.n_vertices, 2)  # punktu koordinātas
        adj_matrices = []
        for k in range(self.dataset_size):
            adj_matrix = np.empty(shape=(self.n_vertices, self.n_vertices))
            for i in range(self.n_vertices):
                for j in range(self.n_vertices):
                    adj_matrix[i][j] = math.sqrt(
                        (coords[k][i][0] - coords[k][j][0]) ** 2 + (coords[k][i][1] - coords[k][j][1]) ** 2)
            adj_matrices.append(adj_matrix.tolist())

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": adj_matrices, "coordinates:": coords})
        data = data.batch(self.batch_size)
        return data

    def validation_data(self) -> tf.data.Dataset:
        return self.train_data()

    def test_data(self) -> tf.data.Dataset:
        return self.test_data()

    def loss(self, predictions, adj_matrix):
        return tsp_loss(predictions, adj_matrix)

    def filter_loss_inputs(self, step_data) -> dict:
        return {"adj_matrix": step_data["adjacency_matrix"]}

    def filter_model_inputs(self, step_data) -> dict:
        return {"inputs": step_data["adjacency_matrix"]}

    def interpret_model_output(self, model_output) -> tf.Tensor:
        return model_output

    def accuracy(self, predictions, step_data):
        return 0., 0.  # tuple: accuracy, total_accuracy
