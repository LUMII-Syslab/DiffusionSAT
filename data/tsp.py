import math
import copy
import random

import numpy as np
import tensorflow as tf

from data.dataset import Dataset

from loss.supervised_tsp import get_path_with_Concorde


class EuclideanTSP(Dataset):

    # TODO(@Emīls): Move batch_size to config and add kwargs to datasets
    def __init__(self, node_count=8, **kwargs) -> None:
        self.node_count = node_count

    def train_data(self) -> tf.data.Dataset:
        data = self.__generate_data(node_count=self.node_count, dataset_size=5000, batch_size=16)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def __generate_data(self, node_count, dataset_size, batch_size) -> tf.data.Dataset:
        # generates 2D Euclidean coordinates and the corresponding adjacency matrices
        coords = np.random.rand(dataset_size, node_count, 2)
        graphs = []
        for u in range(dataset_size):
            graph = np.empty(shape=(node_count, node_count))
            for i in range(node_count):
                for j in range(node_count):
                    graph[i][j] = math.sqrt(
                        (coords[u][i][0] - coords[u][j][0]) ** 2 + (coords[u][i][1] - coords[u][j][1]) ** 2)
            graphs.append(graph.tolist())

        data = tf.data.Dataset.from_tensor_slices({"adjacency_matrix": graphs, "coordinates": coords})
        data = data.batch(batch_size)
        return data

    def validation_data(self) -> tf.data.Dataset:
        data = self.__generate_data(node_count=self.node_count, dataset_size=512, batch_size=64)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def test_data(self) -> tf.data.Dataset:
        data = self.__generate_data(node_count=self.node_count, dataset_size=512, batch_size=64)
        data = data.shuffle(10000)
        data = data.repeat()
        return data

    def filter_model_inputs(self, step_data) -> dict:
        return {"adj_matrix": step_data["adjacency_matrix"], "coords": step_data["coordinates"]}

    def accuracy(self, predictions, step_data):
        # calculates two accuracy metrics:
        # 1) optimality gap of the greedy_search(model(graph))
        # 2) optimality gap of the greedy_search(graph)
        # if the model helps, 1 should be larger than 2

        # getting numpy arrays of predictions, coordinates and adjacency matrices
        batch_size, node_count, *_ = tf.shape(predictions)
        predictions_reshaped = tf.reshape(predictions, shape=[batch_size, node_count, node_count])
        coordinates_reshaped = tf.reshape(step_data["coordinates"], shape=[batch_size, node_count, 2])
        adjacency_matrices_reshaped = tf.reshape(step_data["adjacency_matrix"], shape=[batch_size, node_count, node_count])
        predictions_np = copy.deepcopy(predictions_reshaped.numpy())
        coordinates_np = copy.deepcopy(coordinates_reshaped.numpy())
        adjacency_matrices_np = copy.deepcopy(adjacency_matrices_reshaped.numpy())

        raw_optimal_path_len_sum = 0
        raw_greedy_path_len_sum = 0
        model_greedy_path_len_sum = 0
        # raw_beam_path_len_sum = 0
        # model_beam_path_len_sum = 0

        for i in range(len(adjacency_matrices_np)):  # iterate over the batch
            raw_optimal_path = get_path_with_Concorde(coordinates_np[i])
            raw_optimal_path_len = get_path_len(adjacency_matrices_np[i], raw_optimal_path)
            raw_optimal_path_len_sum += raw_optimal_path_len

            raw_greedy_path = get_path_from_score_greedy(adjacency_matrices_np[i], shortest=True)
            raw_greedy_path_len = get_path_len(adjacency_matrices_np[i], raw_greedy_path)
            raw_greedy_path_len_sum += raw_greedy_path_len

            model_greedy_path = get_path_from_score_greedy(predictions_np[i])
            model_greedy_path_len = get_path_len(adjacency_matrices_np[i], model_greedy_path)
            model_greedy_path_len_sum += model_greedy_path_len

            # raw_beam_path = get_path_from_score_beam(adjacency_matrices_np[i], shortest=True)
            # raw_beam_path_len = get_path_len(adjacency_matrices_np[i], raw_beam_path)
            # raw_beam_path_len_sum += raw_beam_path_len
            #
            # model_beam_path = get_path_from_score_beam(predictions_np[i])
            # model_beam_path_len = get_path_len(adjacency_matrices_np[i], model_beam_path)
            # model_beam_path_len_sum += model_beam_path_len

        model_greedy_optimality_gap = raw_optimal_path_len_sum / model_greedy_path_len_sum
        raw_greedy_optimality_gap = raw_optimal_path_len_sum / raw_greedy_path_len_sum
        return model_greedy_optimality_gap, raw_greedy_optimality_gap

    # def visualize_TSP(self, predictions, step_data):
    #     # getting numpy arrays of predictions, coordinates and adjacency matrices
    #     batch_size, node_count, *_ = tf.shape(predictions)
    #     predictions_reshaped = tf.reshape(predictions, shape=[batch_size, node_count, node_count])
    #     coordinates_reshaped = tf.reshape(step_data["coordinates"], shape=[batch_size, node_count, 2])
    #     predictions_np = copy.deepcopy(predictions_reshaped.numpy())
    #     coordinates_np = copy.deepcopy(coordinates_reshaped.numpy())
    #
    #     random_index = np.random.randint(low=0, high=batch_size, size=2)[0]
    #     draw_graph(predictions_np[random_index], coordinates_np[random_index], "pink")




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
    for i in range(1,len(score)):
        current_vertex = argrandom(score[current_vertex], path)
        path.append(current_vertex)
    path.append(0)
    return path

def get_path_from_score_greedy(score, shortest=False):
    # Greedily picking the best edge to unvisited vertex
    if(shortest==True): argopt=argmin
    else: argopt=argmax
    path = []
    current_vertex = 0
    path.append(current_vertex)
    for i in range(1,len(score)):
        current_vertex = argopt(score[current_vertex], path)
        path.append(current_vertex)
    path.append(0)
    return path

def get_path_from_score_beam(score, beam_size=50, branch_factor=4, shortest=False):
    # Beam search the best edge to unvisited vertex
    """
    score (2D np array) - a input adjacency matrix
    beam_size (int) - the number of simultaneously considered paths (size when pruned)
    branch_factor (int) - number of continuations considered for each path
    shortest (bool) - True if searching for the shortest path, False if for the longest
    """

    if(shortest): Sort_Pairs = Sort_Pairs_Shortest
    else: Sort_Pairs = Sort_Pairs_Longest

    # create path starting from 0. pair (path, score_sum)
    paths = []
    paths.append([[0], 0])

    for k in range(1, len(score)):  # adding vertices one by one
        current_n_paths = copy.deepcopy(len(paths))
        for i in range(current_n_paths):  # iterating over all beam_size paths

            # best next vertices from the last vertex of current path
            best_next_pairs = []
            for v in range(len(score)):  # for all possible next vertices
                if(v not in paths[i][0]):  # that are not already in path
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
        paths = paths[-beam_size:]  # takes the best paths

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

def get_path_len(adj_matrix, path):
    path_len = 0
    for i in range(len(path)-1):
        path_len += adj_matrix[path[i],path[i+1]]
    return path_len

# def draw_graph(x, coords, colour="cyan"):
#     if(colour == "pink"):
#         red = 1.
#         green = 0.
#     else:
#         red = 0.
#         green = 1.
#     min_val = np.amin(x)
#     max_val = np.amax(x)
#     x = x - min_val  # make it start at 0
#     x = x/(max_val-min_val)  # norm it to range [0-1]
#     n_vertices = len(coords)
#     # x are adjacency matrix, coords are the coordinates of the vertices
#     plt.scatter(coords[:, 0], coords[:, 1], color='k')
#     for i in range(n_vertices):
#         for j in range(i):
#             both_edges = max(x[i][j], x[j][i])
#             one_edge = min(x[i][j], x[j][i])
#             ratio = max(x[i][j], x[j][i])
#             color = (red, green, ratio)  # red, if one-directional edge, pink if both directions
#             plt.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]], alpha=both_edges, color=color, lw='3')
#     plt.show()