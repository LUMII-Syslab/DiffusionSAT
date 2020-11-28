import math
import copy
import random

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

from loss.supervised_tsp import get_path_with_Concorde
from loss.supervised_tsp import get_score_with_Concorde


class EuclideanTSP(Dataset):

    # TODO(@Emīls): Move batch_size to config and add kwargs to datasets
    def __init__(self, node_count=16, **kwargs) -> None:
        self.node_count = node_count
        self.no_labels = False

    def train_data(self) -> tf.data.Dataset:
        data = self.__generate_data(node_count=self.node_count, dataset_size=100000, batch_size=32)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def __generate_data(self, node_count, dataset_size, batch_size) -> tf.data.Dataset:
        # generates 2D Euclidean coordinates and the corresponding adjacency matrices
        coords = np.random.rand(dataset_size, node_count, 2)
        graphs = []
        labels = []
        print_iteration = 1000
        for u in range(dataset_size):
            graph = np.empty(shape=(node_count, node_count))
            for i in range(node_count):
                for j in range(node_count):
                    graph[i][j] = math.sqrt(
                        (coords[u][i][0] - coords[u][j][0]) ** 2 + (coords[u][i][1] - coords[u][j][1]) ** 2)
            if self.no_labels:
                label = []
            else:
                label = get_score_with_Concorde(coords[u])
                labels.append(label.tolist())
            graphs.append(graph.tolist())
            if(u%print_iteration==0):
                print(f"Building TSP dataset: {u}/{dataset_size} done")

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates": coords, "labels": labels})
        data = data.batch(batch_size)
        return data

    def validation_data(self) -> tf.data.Dataset:
        data = self.__generate_data(node_count=self.node_count, dataset_size=1024, batch_size=2)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def test_data(self) -> tf.data.Dataset:
        data = self.__generate_data(node_count=self.node_count, dataset_size=1024, batch_size=16)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def filter_model_inputs(self, step_data) -> dict:
        return {"adj_matrix": step_data["adjacency_matrix"], "coords": step_data["coordinates"], "labels": step_data["labels"]}

    def accuracy(self, predictions, step_data):
        # calculates (optimal_path_length / found_path_length)
        # where found_path_length = path_length(greedy_search(model(graph)))

        batch_size, node_count, *_ = tf.shape(predictions)
        predictions = tf.reshape(predictions, shape=[batch_size, node_count, node_count])
        coordinates = tf.reshape(step_data["coordinates"], shape=[batch_size, node_count, 2])

        input_optimal_path_len_sum = 0
        model_greedy_path_len_sum = 0

        for i in range(batch_size):
            input_optimal_path = get_path_with_Concorde(coordinates[i])
            input_optimal_path_len = get_path_len(coordinates[i], input_optimal_path)
            input_optimal_path_len_sum += input_optimal_path_len

            model_greedy_path = get_path_from_score_greedy(predictions[i])
            model_greedy_path_len = get_path_len(coordinates[i], model_greedy_path)
            model_greedy_path_len_sum += model_greedy_path_len

        model_greedy_accuracy = input_optimal_path_len_sum / model_greedy_path_len_sum
        return model_greedy_accuracy, 0

    def TSP_metrics(self, predictions, step_data):
        # calculates several accuracy metrics:
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

        batch_size, node_count, *_ = tf.shape(predictions)
        predictions_reshaped = tf.reshape(predictions, shape=[batch_size, node_count, node_count])
        coordinates_reshaped = tf.reshape(step_data["coordinates"], shape=[batch_size, node_count, 2])
        adjacency_matrices_reshaped = tf.reshape(step_data["adjacency_matrix"], shape=[batch_size, node_count, node_count])

        # beam search is slow without .numpy()
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
            input_optimal_path = get_path_with_Concorde(coordinates_np[i])
            input_optimal_path_len = get_path_len(coordinates_np[i], input_optimal_path)
            input_optimal_path_len_sum += input_optimal_path_len

            input_greedy_path = get_path_from_score_greedy(adjacency_matrices_np[i], shortest=True)
            input_greedy_path_len = get_path_len(coordinates_np[i], input_greedy_path)
            input_greedy_path_len_sum += input_greedy_path_len

            model_greedy_path = get_path_from_score_greedy(predictions_np[i])
            model_greedy_path_len = get_path_len(coordinates_np[i], model_greedy_path)
            model_greedy_path_len_sum += model_greedy_path_len

            input_beam_path = get_path_from_score_beam(adjacency_matrices_np[i], shortest=True, beam_width=beam_width)
            input_beam_path_len = get_path_len(coordinates_np[i], input_beam_path)
            input_beam_path_len_sum += input_beam_path_len

            model_beam_path = get_path_from_score_beam(predictions_np[i], beam_width=beam_width)
            model_beam_path_len = get_path_len(coordinates_np[i], model_beam_path)
            model_beam_path_len_sum += model_beam_path_len

            input_random_path = get_path_from_score_random(adjacency_matrices_np[i])
            input_random_path_len = get_path_len(coordinates_np[i], input_random_path)
            input_random_path_len_sum += input_random_path_len

        model_greedy_optimality_gap = 100*(model_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        input_greedy_optimality_gap = 100*(input_greedy_path_len_sum / input_optimal_path_len_sum - 1)
        model_beam_optimality_gap = 100*(model_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_beam_optimality_gap = 100*(input_beam_path_len_sum / input_optimal_path_len_sum - 1)
        input_random_optimality_gap = 100*(input_random_path_len_sum / input_optimal_path_len_sum - 1)
        return model_greedy_optimality_gap, input_greedy_optimality_gap, model_beam_optimality_gap, input_beam_optimality_gap, input_random_optimality_gap



def sparsify(graph, percentile_closest=20):
    # currently not functioning with Concorde, as it gets penalized for picking seemingly bad edges
    excluded_edge_value = 3
    percentiles = np.percentile(graph, percentile_closest, axis=1)
    for i in range(len(graph)):
        for j in range(len(graph)):
            if graph[i][j] > percentiles[i]:
                graph[i][j] = excluded_edge_value
                graph[j][i] = excluded_edge_value
    return graph


def argmax(array, excluded):
    val_max = -10000  # non_edge
    index_max = 0
    for i, el in enumerate(array):
        if i not in excluded:
            if el > val_max:
                val_max = el
                index_max = i
    return index_max


def argmin(array, excluded):
    val_min = 10000  # non_edge
    index_min = 0
    for i, el in enumerate(array):
        if i not in excluded:
            if el < val_min:
                val_min = el
                index_min = i
    return index_min


def argrandom(array, excluded):
    candidate_indices = []
    for i in range(len(array)):
        if i not in excluded:
            candidate_indices.append(i)
    return random.choice(candidate_indices)


def get_path_from_score_random(score):
    # Randomly picking the best edge to unvisited vertex
    path = []
    current_vertex = 0
    path.append(current_vertex)
    for i in range(1, len(score)):
        current_vertex = argrandom(score[current_vertex], path)
        path.append(current_vertex)
    path.append(0)
    return path


def get_path_from_score_greedy(score, shortest=False):
    # Greedily picking the best edge to unvisited vertex
    if (shortest == True):
        argopt = argmin
    else:
        argopt = argmax
    path = []
    current_vertex = 0
    path.append(current_vertex)
    for i in range(1, len(score)):
        current_vertex = argopt(score[current_vertex], path)
        path.append(current_vertex)
    path.append(0)
    return path


def get_path_from_score_beam(score, beam_width=50, branch_factor=None, shortest=False):
    # Beam search the best edge to unvisited vertex
    """
    score (2D np array) - a input adjacency matrix
    beam_size (int) - the number of simultaneously considered paths (size when pruned)
    branch_factor (int) - number of continuations considered for each path
    shortest (bool) - True if searching for the shortest path, False if for the longest
    """

    if (branch_factor == None):
        branch_factor = len(score)

    if (shortest):
        Sort_Pairs = Sort_Pairs_Shortest
    else:
        Sort_Pairs = Sort_Pairs_Longest

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
            best_next_pairs = Sort_Pairs(best_next_pairs)
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
        paths = Sort_Pairs(paths)  # sort by score_sum
        paths = paths[-beam_width:]  # takes the best paths

    # add the last edge to 0 to all paths
    for i in range(len(paths)):  # iterating over all beam_size paths
        v = 0
        new_pair = copy.deepcopy(paths[i])
        new_pair[1] += score[new_pair[0][-1], v]  # old sum + edge
        new_pair[0].append(v)
        paths.append(new_pair)
    paths = paths[current_n_paths:]  # delete the paths with no continuation
    paths = Sort_Pairs(paths)
    path = paths[-1][0]  # takes the best path

    return path


def Sort_Pairs_Longest(list_of_pairs):
    # sort a list of lists by the second Item
    # returns the pairs sorted in ascending order
    return (sorted(list_of_pairs, key=lambda x: x[1]))


def Sort_Pairs_Shortest(list_of_pairs):
    # sort a list of lists by the second Item
    # returns the pairs sorted in descending order
    return (sorted(list_of_pairs, key=lambda x: -x[1]))


# def get_path_len(adj_matrix, path):
#     path_len = 0
#     for i in range(len(path) - 1):
#         path_len += adj_matrix[path[i], path[i + 1]]
#     return path_len

def get_path_len(coords, path):
    path_len = 0
    for k in range(len(path) - 1):
        i = path[k]  # vertex from
        j = path[k + 1]  # vertex to
        path_len += math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
    return path_len