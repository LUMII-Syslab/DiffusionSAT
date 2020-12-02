#todo jāpieliek norm komentāri

cpdef list subtours(int batch_size, int node_count, predictions):
    # variable type definitions for cython
    cdef int g, i, j, x, subtours_added, endpoint1, endpoint2
    cdef double cut_weight, edge1, edge2
    cdef (double, (int, int)) edge
    cdef list subtours, sorted_edges, subtour_edges, components, subtour_edges_check

    subtours = []
    subtours_added = 0
    for g in range(batch_size):
        graph = predictions[g]
        sorted_edges = []

        for i in range(node_count):
            for j in range(node_count):
                edge1 = graph[i][j]
                edge2 = graph[j][i]
                sorted_edges.append((edge1 + edge2, (i, j)))
        sorted_edges.sort(reverse=True)

        subtour_edges = []
        components = list(range(node_count))
        for edge in sorted_edges:
            endpoint1 = edge[1][0]  # šķautnes virsotnes
            endpoint2 = edge[1][1]

            if components[endpoint1] != components[endpoint2]:
                # šis notiks O(n) reizes:
                components[:] = [components[endpoint1] if x == components[endpoint2] else x for x in components]

                edge_component = [x for x in range(node_count) if components[x] == components[endpoint1]]
                edge_component = set(edge_component)  # kopa ar virsotnēm no tās komponentes, kurā ir edge

                if len(edge_component) == node_count:
                    break

                cut_weight = 0
                subtour_check = []
                for i in range(node_count):
                    for j in range(node_count):
                        if (i in edge_component) ^ (j in edge_component):
                            cut_weight += graph[i][j]
                            subtour_check.append([subtours_added, g * node_count + node_count * i + j])

                if cut_weight < 2:
                    subtours_added += 1
                    subtours.extend(subtour_check)

    return subtours  # atgriež list ar subtours - vnk ind prieks lielā sparse tensora, kur katrā rindiņā būs konkrēts pārkāpts subtour
