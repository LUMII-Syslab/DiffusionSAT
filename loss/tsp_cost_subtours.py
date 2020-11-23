import copy
import tensorflow as tf

#TODO(@Elīza): šajā failā atstāt visu, kas nesaistās ar tensorflow
def subtour_constraints(predictions_tensor):
    batch_size, node_count, *_ = tf.shape(predictions_tensor)
    predictions = predictions_tensor.numpy()

    # TODO(@Elīza): Optimize this, can something be cached or precomputed, can we parallelize it along batch dimension?

    subtours = []

    for g in range(len(predictions)):
        graph = predictions[g]
        sorted_edges = []
        for i in range(node_count):
            for j in range(node_count):
                sorted_edges.append((graph[i][j] + graph[j][i], (i, j)))
        sorted_edges.sort(reverse=True)

        subtour_edges = []
        subtours_added = 0
        components = list(range(node_count))
        for edge in sorted_edges:  # šis notiks O(n) reizes
            endpoint1 = edge[1][0]  # šķautnes virsotnes
            endpoint2 = edge[1][1]

            if components[endpoint1] != components[endpoint2]:
                components[:] = [components[endpoint1] if x == components[endpoint2] else x for x in components]

                edge_component = [x for x in range(len(components)) if x != endpoint1 and
                                                                       x != endpoint2 and
                                                                       components[x] == components[endpoint1]]
                edge_component.extend([endpoint1, endpoint2])
                edge_component = set(edge_component)  # kopa ar virsotnēs no tās komponentes, kurā ir edge

                if len(edge_component) == node_count:
                    break

                cut_weight = 0
                subtour_edges_check = []
                for i in range(node_count):
                    for j in range(node_count):
                        if (i in edge_component) ^ (j in edge_component):
                            cut_weight += graph[i][j]
                            subtour_edges_check.append([subtours_added, node_count * i + j])

                if cut_weight < 2:
                    subtour_edges.extend(subtour_edges_check)
                    subtours_added += 1

        if subtours_added != 0:
            subtour = tf.SparseTensor(values=[1.] * len(subtour_edges), indices=subtour_edges,
                                dense_shape=[subtours_added, node_count * node_count])
            subtours.append((g, subtour))

    return subtours
