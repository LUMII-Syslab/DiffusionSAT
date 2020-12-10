import copy
import math
import os
import random

import numpy as np
import tensorflow as tf
from concorde.tsp import TSPSolver

from metrics.base import Metric

PADDING_VALUE = -1


class TSPAccuracy(Metric):
    def __init__(self) -> None:
        self.mean_acc = tf.metrics.Mean()

    def update_state(self, model_output, step_data):
        acc = self.__accuracy(model_output["prediction"], step_data)
        self.mean_acc.update_state(acc)

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        mean_acc = self.mean_acc.result()
        if reset_state:
            self.reset_state()

        with tf.name_scope("accuracy"):
            tf.summary.scalar("accuracy", mean_acc, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        mean_acc = self.mean_acc.result().numpy()
        if reset_state:
            self.reset_state()

        print(f"Accuracy: {mean_acc:.4f}")

    def reset_state(self):
        self.mean_acc.reset_states()

    @staticmethod
    def __accuracy(predictions, step_data):
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
        return model_greedy_accuracy


class TSPMetrics(Metric):

    def __init__(self) -> None:
        self.mean_model_greedy = tf.metrics.Mean()
        self.mean_input_greedy = tf.metrics.Mean()
        self.mean_model_beam = tf.metrics.Mean()
        self.mean_input_beam = tf.metrics.Mean()
        self.mean_input_random = tf.metrics.Mean()

    def update_state(self, model_output, step_data):
        results = self.__calculate_metrics(model_output["prediction"], step_data)
        self.mean_model_greedy.update_state(results[0])
        self.mean_input_greedy.update_state(results[1])
        self.mean_model_beam.update_state(results[2])
        self.mean_input_beam.update_state(results[3])
        self.mean_input_random.update_state(results[4])

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        model_greedy, input_greedy, model_beam, input_beam, input_random = self.__get_scores()

        if reset_state:
            self.reset_state()

        with tf.name_scope("TSP_metrics"):
            tf.summary.scalar("model/greedy", model_greedy, step=step)
            tf.summary.scalar("input/greedy", input_greedy, step=step)
            tf.summary.scalar("model/beam", model_beam, step=step)
            tf.summary.scalar("input/beam", input_beam, step=step)
            tf.summary.scalar("input/random", input_random, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        model_greedy, input_greedy, model_beam, input_beam, input_random = self.__get_scores()

        if reset_state:
            self.reset_state()

        print(f"model_greedy: {model_greedy.numpy():.2f}%; "
              f"input_greedy {input_greedy.numpy():.2f}%; "
              f"model_beam: {model_beam.numpy():.2f}%; "
              f"input_beam: {input_beam.numpy():.2f}%; "
              f"input_random: {input_random.numpy():.2f}%; ")

    def reset_state(self):
        self.mean_model_greedy.reset_states()
        self.mean_input_greedy.reset_states()
        self.mean_model_beam.reset_states()
        self.mean_input_beam.reset_states()
        self.mean_input_random.reset_states()

    def __get_scores(self):
        return (self.mean_model_greedy.result(),
                self.mean_input_greedy.result(),
                self.mean_model_beam.result(),
                self.mean_input_beam.result(),
                self.mean_input_random.result())

    @staticmethod
    def __calculate_metrics(predictions, step_data):
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
        adjacency_matrices_reshaped = tf.reshape(step_data["adjacency_matrix"],
                                                 shape=[batch_size, padded_size, padded_size])

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

        model_greedy_optimality_gap = 100 * (model_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        input_greedy_optimality_gap = 100 * (input_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        model_beam_optimality_gap = 100 * (model_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_beam_optimality_gap = 100 * (input_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_random_optimality_gap = 100 * (input_random_path_len_sum / input_optimal_path_len_sum - 1)
        return model_greedy_optimality_gap, input_greedy_optimality_gap, model_beam_optimality_gap, input_beam_optimality_gap, input_random_optimality_gap


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
        # passing the x and y coordinates to the Concorde solver to find optimal Euclidean 2D TSP:
        solver = TSPSolver.from_data(coordinates[:, 0], coordinates[:, 1], norm="GEO")
        solution = solver.solve()
    # Only write instances with valid solutions
    assert (np.sort(solution.tour) == np.arange(len(coordinates))).all()  # all nodes are in the solution
    path = np.append(solution.tour, [solution.tour[0]])  # return to the first node
    return path
