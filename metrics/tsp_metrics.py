import copy
import math
import random

import tensorflow as tf

from metrics.base import Metric

PADDING_VALUE = -1


class TSPInitialMetrics(Metric):
    # calculates the optimality gap of greedy, beam and random search on the input graph.
    def __init__(self, beam_width) -> None:
        self.mean_input_greedy = tf.metrics.Mean()
        self.mean_input_beam = tf.metrics.Mean()
        self.mean_input_random = tf.metrics.Mean()
        self.beam_width = beam_width

    def update_state(self, model_output, step_data):
        results = self.__calculate_metrics(model_output["prediction"], step_data, self.beam_width)
        self.mean_input_greedy.update_state(results[0])
        self.mean_input_beam.update_state(results[1])
        self.mean_input_random.update_state(results[2])

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        input_greedy, input_beam, input_random = self.__get_scores()

        if reset_state:
            self.reset_state()

        with tf.name_scope("TSP_initial_metrics"):
            tf.summary.scalar("input/greedy", input_greedy, step=step)
            tf.summary.scalar("input/beam", input_beam, step=step)
            tf.summary.scalar("input/random", input_random, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        input_greedy, input_beam, input_random = self.__get_scores()

        if reset_state:
            self.reset_state()

        print(f"Initial metrics: "
              f"input_greedy {input_greedy.numpy():.4f}%; "
              f"input_beam: {input_beam.numpy():.4f}%; "
              f"input_random: {input_random.numpy():.4f}%")

    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        pass

    def reset_state(self):
        self.mean_input_greedy.reset_states()
        self.mean_input_beam.reset_states()
        self.mean_input_random.reset_states()

    def __get_scores(self):
        return (self.mean_input_greedy.result(),
                self.mean_input_beam.result(),
                self.mean_input_random.result())

    @staticmethod
    def __calculate_metrics(predictions, step_data, beam_width):
        # calculates several TSP accuracy metrics:
        # 1) average optimality gap of the greedy_search(graph)
        # 2) average optimality gap of the beam_search(graph)
        # 3) average optimality gap of the random_search(graph)
        # average optimality gap is defined as: AVERAGE (found_path_length / optimal_path_length - 1)
        # It is reported in percents. Smaller optimality gap is better.

        batch_size, padded_size, *_ = tf.shape(predictions)
        labels = tf.reshape(step_data["labels"], shape=[batch_size, padded_size, padded_size]).numpy()
        adjacency_matrices = tf.reshape(step_data["adjacency_matrix"], shape=[batch_size, padded_size, padded_size]).numpy()

        input_optimal_path_len_sum = 0
        input_greedy_path_len_sum = 0
        input_beam_path_len_sum = 0
        input_random_path_len_sum = 0

        for i in range(batch_size):
            node_count = get_unpadded_size(labels[i])
            label = remove_padding(labels[i], unpadded_size=node_count)
            adjacency_matrix = remove_padding(adjacency_matrices[i], unpadded_size=node_count)

            input_optimal_path = get_path_from_score_greedy(label)
            input_optimal_path_len = get_path_len_from_adj_matrix(adjacency_matrix, input_optimal_path)
            input_optimal_path_len_sum += input_optimal_path_len

            input_greedy_path = get_path_from_score_greedy(adjacency_matrix, shortest=True)
            input_greedy_path_len = get_path_len_from_adj_matrix(adjacency_matrix, input_greedy_path)
            input_greedy_path_len_sum += input_greedy_path_len

            input_beam_path = get_path_from_score_beam(adjacency_matrix, shortest=True, beam_width=beam_width)
            input_beam_path_len = get_path_len_from_adj_matrix(adjacency_matrix, input_beam_path)
            input_beam_path_len_sum += input_beam_path_len

            input_random_path = get_path_from_score_random(adjacency_matrix)
            input_random_path_len = get_path_len_from_adj_matrix(adjacency_matrix, input_random_path)
            input_random_path_len_sum += input_random_path_len

        input_greedy_optimality_gap = 100 * (input_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        input_beam_optimality_gap = 100 * (input_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_random_optimality_gap = 100 * (input_random_path_len_sum / input_optimal_path_len_sum - 1)
        return input_greedy_optimality_gap, input_beam_optimality_gap, input_random_optimality_gap


class TSPMetrics(Metric):
    # calculates the optimality gap of greedy and sph beam search on the predictions.
    def __init__(self, beam_width) -> None:
        self.mean_model_greedy = tf.metrics.Mean()
        self.mean_model_beam = tf.metrics.Mean()
        self.beam_width = beam_width

    def update_state(self, model_output, step_data):
        results = self.__calculate_metrics(model_output["prediction"], step_data, self.beam_width)
        self.mean_model_greedy.update_state(results[0])
        self.mean_model_beam.update_state(results[1])

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        model_greedy, model_beam = self.__get_scores()

        if reset_state:
            self.reset_state()

        with tf.name_scope("TSP_metrics"):
            tf.summary.scalar("model/greedy", model_greedy, step=step)
            tf.summary.scalar("model/beam_sph", model_beam, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        model_greedy, model_beam = self.__get_scores()

        if reset_state:
            self.reset_state()

        print(f"model_greedy: {model_greedy.numpy():.4f}%; "
              f"model_beam_sph: {model_beam.numpy():.4f}%")

    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        pass

    def reset_state(self):
        self.mean_model_greedy.reset_states()
        self.mean_model_beam.reset_states()

    def __get_scores(self):
        return (self.mean_model_greedy.result(),
                self.mean_model_beam.result())

    @staticmethod
    def __calculate_metrics(predictions, step_data, beam_width):
        # calculates several TSP accuracy metrics:
        # 1) average optimality gap of the greedy_search(model(graph))
        # 2) average optimality gap of the beam_search(model(graph))
        # average optimality gap is defined as: AVERAGE (found_path_length / optimal_path_length - 1)
        # It is reported in percents. Smaller optimality gap is better.

        batch_size, padded_size, *_ = tf.shape(predictions)
        predictions = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size]).numpy()
        labels = tf.reshape(step_data["labels"], shape=[batch_size, padded_size, padded_size]).numpy()
        adjacency_matrices = tf.reshape(step_data["adjacency_matrix"], shape=[batch_size, padded_size, padded_size]).numpy()

        input_optimal_path_len_sum = 0
        model_greedy_path_len_sum = 0
        model_beam_path_len_sum = 0

        for i in range(batch_size):
            node_count = get_unpadded_size(labels[i])
            prediction = remove_padding(predictions[i], unpadded_size=node_count)
            label = remove_padding(labels[i], unpadded_size=node_count)
            adjacency_matrix = remove_padding(adjacency_matrices[i], unpadded_size=node_count)

            input_optimal_path = get_path_from_score_greedy(label)
            input_optimal_path_len = get_path_len_from_adj_matrix(adjacency_matrix, input_optimal_path)
            input_optimal_path_len_sum += input_optimal_path_len

            model_greedy_path = get_path_from_score_greedy(prediction)
            model_greedy_path_len = get_path_len_from_adj_matrix(adjacency_matrix, model_greedy_path)
            model_greedy_path_len_sum += model_greedy_path_len

            # model_beam_path = get_path_from_score_beam(prediction, beam_width=beam_width)
            model_beam_path = get_path_from_score_beam_sph(prediction, adjacency_matrix, beam_width=beam_width)
            model_beam_path_len = get_path_len_from_adj_matrix(adjacency_matrix, model_beam_path)
            model_beam_path_len_sum += model_beam_path_len

        model_greedy_optimality_gap = 100 * (model_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        model_beam_optimality_gap = 100 * (model_beam_path_len_sum / input_optimal_path_len_sum - 1)
        return model_greedy_optimality_gap, model_beam_optimality_gap


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

    current_n_paths = 0
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

    current_n_paths = copy.deepcopy(len(paths))
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


def get_path_from_score_beam_sph(score, adj_matrix, beam_width=128, branch_factor=None, shortest=False):
    """
    Picking the next edges with beam search to unvisited vertices. Uses shortest path heuristic
    that selects the shortest (calculated from adj_matrix) path from the last beam_width paths.
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

    current_n_paths = 0
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
        if k==len(score)-1:
            break
        paths = sort_pairs(paths)  # sort by score_sum
        paths = paths[-beam_width:]  # takes the best paths

    # add the last edge to 0 to all paths
    current_n_paths = copy.deepcopy(len(paths))
    for i in range(len(paths)):  # iterating over all paths
        v = 0
        new_pair = copy.deepcopy(paths[i])
        new_pair[1] += score[new_pair[0][-1], v]  # old sum + edge
        new_pair[0].append(v)
        paths.append(new_pair)
    paths = paths[current_n_paths:]  # delete the paths with no continuation
    paths = get_real_lengths(paths, adj_matrix)
    paths = sort_pairs_shortest(paths)
    path = paths[-1][0]  # takes the best path

    return path


def get_real_lengths(paths, adj_matrix):
    # calculates the lengths of paths from the adjacency matrix
    n_paths = copy.deepcopy(len(paths))
    for i in range(len(paths)):  # iterating over all beam_size paths
        new_pair = copy.deepcopy(paths[i])
        new_pair[1] = get_path_len_from_adj_matrix(adj_matrix, new_pair[0])
        paths.append(new_pair)
    paths = paths[n_paths:]  # leave only the updated pairs
    return paths


def sort_pairs_longest(list_of_pairs):
    # sort a list of pairs by the second Item
    # returns the pairs sorted in ascending order
    return (sorted(list_of_pairs, key=lambda x: x[1]))


def sort_pairs_shortest(list_of_pairs):
    # sort a list of pairs by the second Item
    # returns the pairs sorted in descending order
    return (sorted(list_of_pairs, key=lambda x: -x[1]))


def get_path_len_from_coords(coords, path):
    # get the Euclidean length of a path through vertices implied by the coordinates
    path_len = 0.0
    for k in range(len(path) - 1):
        i = path[k]  # vertex from
        j = path[k + 1]  # vertex to
        path_len += math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
    return path_len


def get_path_len_from_adj_matrix(adj_matrix, path):
    # get the Euclidean length of a path through vertices; calculated from adjacency matrix
    path_len = 0.0
    for k in range(len(path) - 1):
        i = path[k]  # vertex from
        j = path[k + 1]  # vertex to
        path_len += adj_matrix[i][j]
    return path_len
