import math
import os

import numpy as np
import tensorflow as tf
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde
from pathlib import Path
import shutil

from data.dataset import Dataset
from metrics.tsp_metrics import TSPMetrics, TSPInitialMetrics, PADDING_VALUE


class EuclideanTSP(Dataset):
    # TODO(@Emīls): Move batch_size to config and add kwargs to datasets
    def __init__(self, data_dir, force_data_gen, **kwargs) -> None:
        self.min_node_count = 16
        self.max_node_count = 16
        self.train_data_size = 100000
        self.train_batch_size = 16
        self.beam_width = 128

        self.validation_data_size = 10000
        self.validation_batch_size = 100

        self.test_data_size = 10000
        self.test_batch_size = 100
        self.min_test_node_count = 16
        self.max_test_node_count = 16

        self.force_data_gen = force_data_gen
        self.data_dir = Path(data_dir) / self.__class__.__name__

    def train_data(self) -> tf.data.Dataset:
        return self.fetch_dataset("train", self.min_node_count, self.max_node_count, self.train_data_size, self.train_batch_size)

    def validation_data(self) -> tf.data.Dataset:
        return self.fetch_dataset("validation", self.min_test_node_count, self.max_test_node_count, self.validation_data_size, self.validation_batch_size)

    def test_data(self) -> tf.data.Dataset:
        return self.fetch_dataset("test", self.min_test_node_count, self.max_test_node_count, self.test_data_size, self.test_batch_size)

    def fetch_dataset(self, mode, min_node_count, max_node_count, dataset_size, batch_size):
        """" Reads dataset from file; creates the file if it does not exist."""
        data_folder = self.data_dir / f"{mode}_{min_node_count}_{max_node_count}_{dataset_size//1000}K_{batch_size}"
        data_folder_str = str(data_folder.resolve())  # converts path to a string
        print(f"Fetching TSP {mode} dataset (size={dataset_size//1000}K)")
        if mode == "test": print(f"Number of test batches: {math.ceil(dataset_size/batch_size)}")

        if self.force_data_gen and data_folder.exists():
            shutil.rmtree(data_folder)

        if not data_folder.exists():
            dataset = self.__generate_data(min_node_count, max_node_count, dataset_size, batch_size)
            print(f"Saving TSP {mode} dataset")
            tf.data.experimental.save(dataset, data_folder_str)

        # load, prepare and return the dataset:
        # padded_size = get_nearest_power_of_two(max_node_count)
        padded_size = max_node_count
        element_spec = {'adjacency_matrix': tf.TensorSpec(shape=(None, padded_size, padded_size), dtype=tf.float32),
                     'coordinates': tf.TensorSpec(shape=(None, padded_size, 2), dtype=tf.float32),
                     'labels': tf.TensorSpec(shape=(None, padded_size, padded_size), dtype=tf.float32)}
        dataset = tf.data.experimental.load(data_folder_str, element_spec)
        dataset = dataset.shuffle(dataset_size)
        if mode is not "test": dataset = dataset.repeat()
        return dataset


    def __generate_data(self, min_node_count, max_node_count, dataset_size, batch_size) -> tf.data.Dataset:
        """
        Generates 2D Euclidean TSP dataset.
        The dataset consists of randomly generated TSP coordinates and the corresponding adjacency matrices.
        The labels are matrices with the edges in the optimal path marked by ones.
        """
        graphs = []
        coords = []
        labels = []
        print_iteration = 1000
        # padded_size = get_nearest_power_of_two(max_node_count)
        padded_size = max_node_count

        for i in range(dataset_size):
            node_count = np.random.randint(low=min_node_count, high=max_node_count + 1, size=1)[0]
            graph, coord = generate_graph_and_coord(node_count)
            label = get_label_with_Concorde(coord)
            graph, coord, label = add_padding(graph, coord, label, node_count, padded_size)
            graphs.append(graph.tolist())
            coords.append(coord.tolist())
            labels.append(label.tolist())
            if (i % print_iteration == 0):
                print(f"Building TSP dataset: {i}/{dataset_size} done")

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates": coords, "labels": labels})
        data = data.batch(batch_size)
        return data

    def args_for_train_step(self, step_data) -> dict:
        return {"adj_matrix": step_data["adjacency_matrix"], "coords": step_data["coordinates"],
                "labels": step_data["labels"]}

    def metrics(self, initial=False) -> list:
        if initial:
            return [TSPInitialMetrics(beam_width=self.beam_width)]
        else:
            return [TSPMetrics(beam_width=self.beam_width)]


def generate_graph_and_coord(node_count):
    # returns random 2D coordinates and the corresponding adjacency matrix
    coord = np.random.rand(node_count, 2)
    graph = np.empty(shape=(node_count, node_count))
    for i in range(node_count):
        for j in range(node_count):
            graph[i][j] = math.sqrt(
                (coord[i][0] - coord[j][0]) ** 2 + (coord[i][1] - coord[j][1]) ** 2)
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


class suppress_stdout_stderr(object):
    '''
    Used to stop the Concorde solver from printing.
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function. This will not suppress raised exceptions.
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''

    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def get_path_with_Concorde(coordinates):
    # returns the path corresponding to the optimal Euclidean TSP solution found by the Concorde solver
    with suppress_stdout_stderr():  # suppress_stdout_stderr prevents Concorde from printing
        # passing the x and y coordinates to the Concorde solver to find optimal Euclidean 2D TSP
        # multiplied by 10**8 because EUC_2D norm rounds coordinates to integers
        solver = TSPSolver.from_data(10 ** 8 * coordinates[:, 0], 10 ** 8 * coordinates[:, 1], norm="EUC_2D")
        # solver = TSPSolver.from_data(coordinates[:, 0], coordinates[:, 1], norm="GEO")  # used in literature
        solution = solver.solve()
    # Only write instances with valid solutions
    assert (solution.success and solution.found_tour and not solution.hit_timebound)
    assert (np.sort(solution.tour) == np.arange(len(coordinates))).all()  # all nodes are in the solution
    path = np.append(solution.tour, [solution.tour[0]])  # return to the first node
    return path


def get_label_with_Concorde(coordinates):
    # returns a matrix with edges in the optimal path marked by marked_value
    marked_value = 1.0  # 0.5 for compatibility with unsupervised loss
    node_count = len(coordinates)
    assert node_count >= 4  # Concorde takes 4+ vertices
    path = get_path_with_Concorde(coordinates)
    output = np.zeros((node_count, node_count))
    index = path[0]
    import random
    dir = random.random() > 0.5
    for i in range(node_count):
        previous_index = index
        index = path[i + 1]
        if dir:
            output[previous_index, index] = marked_value
        else:
            output[index, previous_index] = marked_value
    return output
