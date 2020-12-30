import cython
import numpy as np
cimport numpy as np
from scipy.sparse.csgraph._traversal import connected_components

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive

cpdef list subtours(int batch_size, int padded_size, np.ndarray[np.float32_t, ndim = 3] predictions,
                    np.ndarray[np.float32_t, ndim = 3] adjacency_matrix, int PADDING_VALUE, str task):

    # variable type definitions for cython
    cdef int g, i, j, x, subtours_added, endpoint1, endpoint2, edge_component_id, other_id, node_count
    cdef double cut_weight1, cut_weight2, edge1, edge2
    cdef (double, int, int) edge
    cdef list subtours, sorted_edges, subtour_check
    cdef np.ndarray[np.int64_t, ndim = 1] components, unpadded_sizes
    cdef np.ndarray[np.int64_t, ndim = 2] component_graph
    cdef bint one_component

    # todo simpler?
    # array of real sizes of graphs in the batch
    unpadded_sizes = np.empty(batch_size, dtype=np.int64)
    for g in range(batch_size):
        row = adjacency_matrix[g][0]
        if row[padded_size-1] != PADDING_VALUE:
            unpadded_sizes[g] = padded_size
        else:
            unpadded_sizes[g] = np.where(row == PADDING_VALUE)[0][0]


    components = np.zeros(padded_size, dtype=np.int64)
    component_graph = np.zeros([padded_size, padded_size], dtype=np.int64)

    subtours = []
    subtours_added = 0
    for g in range(batch_size):
        node_count = unpadded_sizes[g]
        sorted_edges = []

        # sort edges in descending order
        for i in range(node_count):
            for j in range(node_count):
                edge1 = predictions[g,i,j]
                edge2 = predictions[g,j,i]
                if task == 'euclidean_tsp':
                    sorted_edges.append((edge1 + edge2, i, j))
                elif task == 'asymmetric_tsp':
                    sorted_edges.append((edge1, i, j))
        sorted_edges.sort(reverse=True)

        subtour_edges = []
        for i in range(node_count): components[i] = i

        # graph for adding edges one by one and finding components
        if task == 'asymmetric_tsp':
            for i in range(node_count):
                for j in range(node_count):
                        component_graph[i][j] = 0


        for edge in sorted_edges:
            endpoint1 = edge[1]
            endpoint2 = edge[2]

            if components[endpoint1] != components[endpoint2]:
                # add edge and update components
                if task == 'euclidean_tsp':
                    edge_component_id = components[endpoint1]
                    other_id = components[endpoint2]
                    for i in range(node_count):
                        if components[i]==other_id: components[i] = edge_component_id

                elif task == 'asymmetric_tsp':
                    component_graph[endpoint1][endpoint2] = 1
                    # todo veeeery sloooow and baaaad
                    C = connected_components(component_graph[0:node_count, 0:node_count], directed=True, connection='strong')[1]
                    for i in range(node_count):
                        components[i] = C[i]

                one_component = True
                for i in range(node_count):
                    if components[i] != edge_component_id:
                        one_component = False
                        break

                if one_component: break

                cut_weight1 = 0  # check cut in each direction
                cut_weight2 = 0
                for i in range(node_count):
                    for j in range(node_count):
                        if components[i] == edge_component_id and components[j] != edge_component_id:
                            cut_weight1 += predictions[g,i,j]
                        if components[i] != edge_component_id and components[j] == edge_component_id:
                            cut_weight2 += predictions[g,i,j]

                # if subtour constraint is violated, add it
                # violated subtour is added as a line of 0s and some 1s – corresponding to edges which are on the cut
                if cut_weight1 < 1:
                    subtour_check = []
                    for i in range(node_count):
                        for j in range(node_count):
                            if components[i] == edge_component_id and components[j] != edge_component_id:
                                subtour_check.append([subtours_added, g * padded_size * padded_size + i * padded_size + j])
                    subtours.extend(subtour_check)
                    subtours_added += 1

                if cut_weight2 < 1:
                    subtour_check = []
                    for i in range(node_count):
                        for j in range(node_count):
                            if components[i] != edge_component_id and components[j] == edge_component_id:
                                subtour_check.append([subtours_added, g * padded_size * padded_size + i * padded_size + j])
                    subtours.extend(subtour_check)
                    subtours_added += 1

    return subtours
