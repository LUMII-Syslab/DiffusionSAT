import os
import math
import copy
import random

import numpy as np
import tensorflow as tf
from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde

from data.dataset import Dataset

PADDING_VALUE = -1


class EuclideanTSP(Dataset):
    # TODO(@Emīls): Move batch_size to config and add kwargs to datasets
    def __init__(self, **kwargs) -> None:
        self.min_node_count = 16
        self.max_node_count = 16
        self.train_data_size = 10000
        self.train_batch_size = 16

    def train_data(self) -> tf.data.Dataset:
        data = self.__generate_data(self.min_node_count, self.max_node_count, self.train_data_size, self.train_batch_size)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def validation_data(self) -> tf.data.Dataset:
        data = self.__generate_data(self.min_node_count, self.max_node_count, dataset_size=100, batch_size=1)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def test_data(self) -> tf.data.Dataset:
        data = self.__generate_data(self.min_node_count, self.max_node_count, dataset_size=2000, batch_size=32)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

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
        padded_size = get_nearest_power_of_two(max_node_count)

        for i in range(dataset_size):
            node_count = np.random.randint(low=min_node_count, high=max_node_count+1, size=1)[0]
            graph, coord = generate_graph_and_coord(node_count)
            label = get_score_with_Concorde(coord)
            graph, coord, label = add_padding(graph, coord, label, node_count, padded_size)
            graphs.append(graph.tolist())
            coords.append(coord.tolist())
            labels.append(label.tolist())
            if(i % print_iteration == 0):
                print(f"Building TSP dataset: {i}/{dataset_size} done")

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates": coords, "labels": labels})
        data = data.batch(batch_size)
        return data

    def filter_model_inputs(self, step_data) -> dict:
        return {"adj_matrix": step_data["adjacency_matrix"], "coords": step_data["coordinates"], "labels": step_data["labels"]}

    def accuracy(self, predictions, step_data):
        # calculates the average (optimal_path_length / found_path_length) in the batch.
        # where found_path_length = path_length(greedy_search(model(graph)))

        batch_size, padded_size, *_ = tf.shape(predictions)
        predictions = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size])
        coordinates = tf.reshape(step_data["coordinates"], shape=[batch_size, padded_size, 2])

        input_optimal_path_len_sum = 0
        model_greedy_path_len_sum = 0

        for i in range(batch_size):
            node_count = get_unpadded_size(coordinates[i])
            coordinate = remove_padding(coordinates[i], unpadded_size=node_count)
            prediction = remove_padding(predictions[i], unpadded_size=node_count)

            input_optimal_path = get_path_with_Concorde(coordinate)
            input_optimal_path_len = get_path_len(coordinate, input_optimal_path)
            input_optimal_path_len_sum += input_optimal_path_len

            model_greedy_path = get_path_from_score_greedy(prediction)
            model_greedy_path_len = get_path_len(coordinate, model_greedy_path)
            model_greedy_path_len_sum += model_greedy_path_len

        model_greedy_accuracy = input_optimal_path_len_sum / model_greedy_path_len_sum
        return model_greedy_accuracy, 0

    def TSP_metrics(self, predictions, step_data):
        # calculates several TSP accuracy metrics:
        # 1) average optimality gap of the greedy_search(model(graph))
        # 2) average optimality gap of the greedy_search(graph)
        # 3) average optimality gap of the beam_search(model(graph))
        # 4) average optimality gap of the beam_search(graph)
        # 5) average optimality gap of the random_search(graph)

        # average optimality gap is defined as: AVERAGE (found_path_length / optimal_path_length - 1)
        # It is reported in percents. Smaller optimality gap is better.
        # 2, 4, 5 stay constant because those do not involve the learned model.
        # if the model helps greedy search, 1 should be smaller than 2
        # if the model helps beam search, 3 should be smaller than 5

        beam_width = 128

        batch_size, padded_size, *_ = tf.shape(predictions)
        predictions_reshaped = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size])
        coordinates_reshaped = tf.reshape(step_data["coordinates"], shape=[batch_size, padded_size, 2])
        adjacency_matrices_reshaped = tf.reshape(step_data["adjacency_matrix"], shape=[batch_size, padded_size, padded_size])

        # beam search is slower without .numpy()
        predictions_np = predictions_reshaped.numpy()
        coordinates_np = coordinates_reshaped.numpy()
        adjacency_matrices_np = adjacency_matrices_reshaped.numpy()

        input_optimal_path_len_sum = 0
        input_greedy_path_len_sum = 0
        model_greedy_path_len_sum = 0
        input_beam_path_len_sum = 0
        model_beam_path_len_sum = 0
        input_random_path_len_sum = 0

        for i in range(batch_size):
            node_count = get_unpadded_size(coordinates_np[i])
            prediction = remove_padding(predictions_np[i], unpadded_size=node_count)
            coordinate = remove_padding(coordinates_np[i], unpadded_size=node_count)
            adjacency_matrix = remove_padding(adjacency_matrices_np[i], unpadded_size=node_count)

            input_optimal_path = get_path_with_Concorde(coordinate)
            input_optimal_path_len = get_path_len(coordinate, input_optimal_path)
            input_optimal_path_len_sum += input_optimal_path_len

            input_greedy_path = get_path_from_score_greedy(adjacency_matrix, shortest=True)
            input_greedy_path_len = get_path_len(coordinate, input_greedy_path)
            input_greedy_path_len_sum += input_greedy_path_len

            model_greedy_path = get_path_from_score_greedy(prediction)
            model_greedy_path_len = get_path_len(coordinate, model_greedy_path)
            model_greedy_path_len_sum += model_greedy_path_len

            input_beam_path = get_path_from_score_beam(adjacency_matrix, shortest=True, beam_width=beam_width)
            input_beam_path_len = get_path_len(coordinate, input_beam_path)
            input_beam_path_len_sum += input_beam_path_len

            model_beam_path = get_path_from_score_beam(prediction, beam_width=beam_width)
            model_beam_path_len = get_path_len(coordinate, model_beam_path)
            model_beam_path_len_sum += model_beam_path_len

            input_random_path = get_path_from_score_random(adjacency_matrix)
            input_random_path_len = get_path_len(coordinate, input_random_path)
            input_random_path_len_sum += input_random_path_len

        model_greedy_optimality_gap = 100*(model_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        input_greedy_optimality_gap = 100*(input_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        model_beam_optimality_gap = 100*(model_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_beam_optimality_gap = 100*(input_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_random_optimality_gap = 100*(input_random_path_len_sum / input_optimal_path_len_sum - 1)
        return model_greedy_optimality_gap, input_greedy_optimality_gap, model_beam_optimality_gap, input_beam_optimality_gap, input_random_optimality_gap



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


def remove_padding(padded_array, unpadded_size=None):
    # returns the input array without the padding
    if unpadded_size == None:
        unpadded_size = get_unpadded_size(padded_array)
    array = padded_array[:unpadded_size, :unpadded_size]
    return array


def get_unpadded_size(padded_array):
    # determines the size of padded_array's first dimension without the padding
    unpadded_size = 1  # skips the first element - for TSP it is likely 0
    padded_size = len(padded_array)
    for i in range(1, padded_size):  # count non-padded values
        if padded_array[i][0] != PADDING_VALUE:
            unpadded_size += 1
        else:
            break
    return unpadded_size


def get_nearest_power_of_two(integer):
    # returns the smallest power of two that is greater or equal than the input
    result = 1
    while result < integer:
        result *= 2
    return result


def argmax(array, excluded):
    # returns the index of the array's largest element, which is not in the excluded
    val_max = -10000  # non_edge
    index_max = 0
    for i, el in enumerate(array):
        if i not in excluded:
            if el > val_max:
                val_max = el
                index_max = i
    return index_max


def argmin(array, excluded):
    # returns the index of the array's smallest element, which is not in the excluded
    val_min = 10000  # non_edge
    index_min = 0
    for i, el in enumerate(array):
        if i not in excluded:
            if el < val_min:
                val_min = el
                index_min = i
    return index_min


def argrandom(array, excluded):
    # returns a random index of the array's element, which is not in the excluded
    candidate_indices = []
    for i in range(len(array)):
        if i not in excluded:
            candidate_indices.append(i)
    return random.choice(candidate_indices)


def get_path_from_score_random(score):
    # Randomly picking the next edges to unvisited vertices
    path = []
    current_vertex = 0
    path.append(current_vertex)
    for i in range(1, len(score)):
        current_vertex = argrandom(score[current_vertex], path)
        path.append(current_vertex)
    path.append(0)
    return path


def get_path_from_score_greedy(score, shortest=False):
    # Picking the next edges with greedy search to unvisited vertices
    if (shortest == True):  # greedily picks the shortest edges
        argopt = argmin
    else:  # greedily picks the largest edges
        argopt = argmax
    path = []
    current_vertex = 0
    path.append(current_vertex)
    for i in range(1, len(score)):
        current_vertex = argopt(score[current_vertex], path)
        path.append(current_vertex)
    path.append(0)
    return path


def get_path_from_score_beam(score, beam_width=128, branch_factor=None, shortest=False):
    """
    Picking the next edges with beam search to unvisited vertices
    score (2D np array) - input adjacency matrix
    beam_width (int) - the number of simultaneously considered paths (size when pruned)
    branch_factor (int) - number of continuations considered for each path
    shortest (bool) - True if searching for the shortest path, False if for the longest
    """

    if branch_factor is None:  # by default all continuations are considered
        branch_factor = len(score)

    if shortest:
        sort_pairs = sort_pairs_shortest
    else:
        sort_pairs = sort_pairs_longest

    # create path starting from 0. pair (path, score_sum)
    paths = []
    paths.append([[0], 0])

    for k in range(1, len(score)):  # adding vertices one by one
        current_n_paths = copy.deepcopy(len(paths))
        for i in range(current_n_paths):  # iterating over all beam_size paths

            # best next vertices from the last vertex of current path
            best_next_pairs = []
            for v in range(len(score)):  # for all possible next vertices
                if (v not in paths[i][0]):  # that are not already in path
                    best_next_pairs.append((v, score[paths[i][0][-1], v]))  # add to pair candidates
            best_next_pairs = sort_pairs(best_next_pairs)
            best_next_vertices = []
            for j in range(min(branch_factor, len(best_next_pairs))):  # take the best few vertices
                best_next_vertices.append(best_next_pairs[-j][0])

            # add the best path continuations to paths
            for v in best_next_vertices:
                new_pair = copy.deepcopy(paths[i])
                new_pair[1] += score[new_pair[0][-1], v]  # old sum + edge
                new_pair[0].append(v)
                paths.append(new_pair)

        paths = paths[current_n_paths:]  # delete the paths with no continuation
        paths = sort_pairs(paths)  # sort by score_sum
        paths = paths[-beam_width:]  # takes the best paths

    # add the last edge to 0 to all paths
    for i in range(len(paths)):  # iterating over all beam_size paths
        v = 0
        new_pair = copy.deepcopy(paths[i])
        new_pair[1] += score[new_pair[0][-1], v]  # old sum + edge
        new_pair[0].append(v)
        paths.append(new_pair)
    paths = paths[current_n_paths:]  # delete the paths with no continuation
    paths = sort_pairs(paths)
    path = paths[-1][0]  # takes the best path

    return path


def sort_pairs_longest(list_of_pairs):
    # sort a list of pairs by the second Item
    # returns the pairs sorted in ascending order
    return (sorted(list_of_pairs, key=lambda x: x[1]))


def sort_pairs_shortest(list_of_pairs):
    # sort a list of pairs by the second Item
    # returns the pairs sorted in descending order
    return (sorted(list_of_pairs, key=lambda x: -x[1]))


def get_path_len(coords, path):
    # get the Euclidean length of a path through vertices implied by the coordinates
    path_len = 0
    for k in range(len(path) - 1):
        i = path[k]  # vertex from
        j = path[k + 1]  # vertex to
        path_len += math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
    return path_len


class suppress_stdout_stderr(object):
    '''
    Used to stop the Concorde solver from printing.
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function. This will not suppress raised exceptions.
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''
    def __init__(self):
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
    def __enter__(self):
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)
    def __exit__(self, *_):
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def get_path_with_Concorde(coordinates):
    # returns the path corresponding to the optimal Euclidean TSP solution found by the Concorde solver
    with suppress_stdout_stderr():  # suppress_stdout_stderr prevents Concorde from printing
        # passing the x and y coordinates to the Concorde solver to find optimal Euclidean 2D TSP:
        solver = TSPSolver.from_data(coordinates[:, 0], coordinates[:, 1], norm="GEO")
        solution = solver.solve()
    # Only write instances with valid solutions
    assert (np.sort(solution.tour) == np.arange(len(coordinates))).all()  # all nodes are in the solution
    path = np.append(solution.tour, [solution.tour[0]])  # return to the first node
    return path


def get_score_with_Concorde(coordinates):
    # returns a matrix with edges in the optimal path marked by ones
    node_count = len(coordinates)
    assert node_count >= 4  # Concorde takes 4+ vertices
    path = get_path_with_Concorde(coordinates)
    output = np.zeros((node_count, node_count))
    index = path[0]
    for i in range(node_count):
        previous_index = index
        index = path[i+1]
        output[previous_index, index] = 1
        output[index, previous_index] = 1
    return output
