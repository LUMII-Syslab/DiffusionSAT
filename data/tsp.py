import math

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

from loss.tsp import tsp_loss


class EuclideanTSP(Dataset):

    # TODO(@Emīls): Move batch_size to config and add kwargs to datasets
    def __init__(self, n=8, count=1500, batch_size=15) -> None:
        self.n = n  # TODO(@Elīza): Meaningful names for variables
        self.count = count
        self.batch_size = batch_size

    def train_data(self) -> tf.data.Dataset:
        data = self.__generate_data()
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def __generate_data(self) -> tf.data.Dataset:
        # eiklīda attālumi
        coords = np.random.rand(self.count, self.n, 2)  # punktu koordinātas
        graphs = []
        for u in range(self.count):
            graph = np.empty(shape=(self.n, self.n))
            for i in range(self.n):
                for j in range(self.n):
                    graph[i][j] = math.sqrt(
                        (coords[u][i][0] - coords[u][j][0]) ** 2 + (coords[u][i][1] - coords[u][j][1]) ** 2)
            graphs.append(graph.tolist())
        # # random attālumi:
        # graphs = np.random.rand(count, n, n)
        # for i in range(count):
        #     for j in range(n):
        #         graphs[i][j][j] = 0
        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates:": coords})
        data = data.batch(self.batch_size)
        return data

    def validation_data(self) -> tf.data.Dataset:
        return self.train_data()

    def test_data(self) -> tf.data.Dataset:
        return self.test_data()

    def loss(self, predictions, step_data):
        return tsp_loss(predictions, step_data["adjacency_matrix"])

    def filter_model_inputs(self, step_data) -> dict:
        return {"inputs": step_data["adjacency_matrix"]}

    def accuracy(self, predictions, step_data):
        return 0., 0.
