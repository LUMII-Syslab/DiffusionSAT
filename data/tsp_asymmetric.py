import numpy as np
import tensorflow as tf
import math
from pathlib import Path

from data.dataset import Dataset
from loss.unsupervised_tsp import inverse_identity
from metrics.tsp_metrics import TSPMetrics, TSPInitialMetrics, PADDING_VALUE
from data.asymmetric_tsp_gen import generate_asymmetric_tsp


class AsymmetricTSP(Dataset):
    def __init__(self, **kwargs) -> None:
        self.min_node_count = 6
        self.max_node_count = 8
        self.train_data_size = 1000
        self.train_batch_size = 16
        self.beam_width = 128

    def train_data(self) -> tf.data.Dataset:
        # data = self.read_data_from_file(self.min_node_count, self.max_node_count, self.train_data_size,
        #                                 self.train_batch_size) # use this only for unsupervised (no labels, but quicker)
        data = self.read_data_from_file(self.min_node_count, self.max_node_count, self.train_data_size,
                                    self.train_batch_size)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def validation_data(self) -> tf.data.Dataset:
        data = self.read_data_from_file(self.min_node_count, self.max_node_count, dataset_size=100, batch_size=1)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def test_data(self) -> tf.data.Dataset:
        data = self.__generate_data(self.min_node_count, self.max_node_count, dataset_size=2000, batch_size=32)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def read_data_from_file(self, min_node_count, max_node_count, dataset_size, batch_size) -> tf.data.Dataset:
        # reads file made by asymmetric_tsp_gen.py

        graphs = []
        coords = []
        labels = []
        padded_size = get_nearest_power_of_two(max_node_count)

        path = Path("/user/deep_loss/data/asymmetric_tsp_{}-{}_{}.txt".format(min_node_count, max_node_count, dataset_size))
        if not path.is_file():
            generate_asymmetric_tsp(min_node_count, max_node_count, dataset_size)

        with open(path) as f:
            next(f)
            next(f)

            for i in range(dataset_size):
                node_count = int(next(f))

                graph = []
                for j in range(node_count):
                    line = next(f)
                    graph.append([float(x) for x in line.split()])

                label = np.zeros(shape=(node_count, node_count))
                line = next(f).split()
                for j in range(node_count - 1):
                    label[int(line[j]), int(line[j+1])] = 1
                label[int(line[-1]), int(line[0])] = 1

                angle = 0
                coord = np.empty([node_count, 2])
                for j in range(node_count):
                    coord[j, 0] = 0.5 * (1 + math.cos(angle))
                    coord[j, 1] = 0.5 * (1 + math.sin(angle))
                    angle += 2 * math.pi / node_count

                graph, coord, label = add_padding(graph, coord, label, node_count, padded_size)
                graphs.append(graph.tolist())
                coords.append(coord.tolist())
                labels.append(label.tolist())

                _ = next(f)

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates": coords, "labels": labels})
        data = data.batch(batch_size)
        return data


    # use this only for unsupervised (no labels but quicker)
    def __generate_data(self, min_node_count, max_node_count, dataset_size, batch_size) -> tf.data.Dataset:
        graphs = []
        coords = []
        labels = []  # todo empty labels
        print_iteration = 1000
        padded_size = get_nearest_power_of_two(max_node_count)

        for i in range(dataset_size):
            node_count = np.random.randint(low=min_node_count, high=max_node_count+1, size=1)[0]
            graph, coord = generate_graph_and_coord(node_count)
            label = np.empty(shape=(node_count, node_count))
            graph, coord, label = add_padding(graph, coord, label, node_count, padded_size)
            graphs.append(graph.tolist())
            coords.append(coord.tolist())
            labels.append(label.tolist())
            if i % print_iteration == 0:
                print(f"Building Asymmetric TSP dataset: {i}/{dataset_size} done")

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates": coords, "labels": labels})
        data = data.batch(batch_size)
        return data


    def filter_model_inputs(self, step_data) -> dict:
        return {"adj_matrix": step_data["adjacency_matrix"], "coords": step_data["coordinates"], "labels": step_data["labels"]}


    def metrics(self, initial=False) -> list:
        if initial:
            return [TSPInitialMetrics(beam_width=self.beam_width)]
        else:
            return [TSPMetrics(beam_width=self.beam_width)]



def generate_graph_and_coord(node_count):
    # places points in a regular polygon; used for visualizing
    angle = 0
    coord = np.empty([node_count, 2])
    for i in range(node_count):
        coord[i, 0] = 0.5 * (1 + math.cos(angle))
        coord[i, 1] = 0.5 * (1 + math.sin(angle))
        angle += 2 * math.pi / node_count

    graph = np.random.random([node_count, node_count]) * inverse_identity(node_count)

    return graph, coord


def add_padding(graph, coord, label, node_count, padded_size):
    # pads graph, coordinates and label up to padded_size
    padding_size = padded_size - node_count
    padded_graph = np.pad(graph, (0, padding_size), 'constant', constant_values=(PADDING_VALUE))
    padded_coord = np.pad(coord, ((0, padding_size), (0, 0)), 'constant', constant_values=(PADDING_VALUE))
    padded_label = np.pad(label, (0, padding_size), 'constant', constant_values=(PADDING_VALUE))
    return padded_graph, padded_coord, padded_label


def get_nearest_power_of_two(integer):
    # returns the smallest power of two that is greater or equal than the input
    result = 1
    while result < integer:
        result *= 2
    return result

