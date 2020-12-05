#todo jāpieliek norm komentāri
import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False) # compiler directive
@cython.wraparound(False) # compiler directive
cpdef list subtours(int batch_size, int node_count, np.ndarray[np.float32_t, ndim = 3] predictions):
    # variable type definitions for cython
    cdef int g, i, j, x, subtours_added, endpoint1, endpoint2
    cdef double cut_weight, edge1, edge2
    cdef (double, int, int) edge
    cdef list subtours, sorted_edges, subtour_edges, subtour_edges_check
    cdef np.ndarray[np.int64_t, ndim = 1] components
    cdef int edge_component_id, other_id
    cdef bint one_component

    components = np.zeros(node_count, dtype=np.int64)

    subtours = []
    subtours_added = 0
    for g in range(batch_size):
        sorted_edges = []

        for i in range(node_count):
            for j in range(node_count):
                edge1 = predictions[g,i,j]
                edge2 = predictions[g,j,i]
                sorted_edges.append((edge1 + edge2, i, j))
        sorted_edges.sort(reverse=True)

        subtour_edges = []
        for i in range(node_count): components[i] = i

        for edge in sorted_edges:
            endpoint1 = edge[1]  # šķautnes virsotnes
            endpoint2 = edge[2]

            if components[endpoint1] != components[endpoint2]:
                # šis notiks O(n) reizes:
                edge_component_id = components[endpoint1]
                other_id = components[endpoint2]
                for i in range(node_count):
                    if components[i]==other_id: components[i] = edge_component_id

                one_component = True
                for i in range(node_count):
                    if components[i] != edge_component_id:
                        one_component = False
                        break

                if one_component:break

                cut_weight = 0
                for i in range(node_count):
                    for j in range(node_count):
                        if (components[i] == edge_component_id) ^ (components[j] == edge_component_id):
                            cut_weight += predictions[g,i,j]

                if cut_weight < 2:
                    subtour_check = []
                    for i in range(node_count):
                        for j in range(node_count):
                            if (components[i] == edge_component_id) ^ (components[j] == edge_component_id):
                                subtour_check.append([subtours_added, g * node_count * node_count + i * node_count + j])

                    subtours_added += 1
                    subtours.extend(subtour_check)

    return subtours  # atgriež list ar subtours - vnk ind prieks lielā sparse tensora, kur katrā rindiņā būs konkrēts pārkāpts subtour
