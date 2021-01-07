import random
import numpy as np
import itertools


def inverse_identity(size):
    return np.ones(shape=[size, size]) - np.eye(size)


# Generate a file of asymmetric tsp graphs with labels
def generate_asymmetric_tsp(min_node_count, max_node_count, dataset_size):
    name = "/user/deep_loss/data/asymmetric_tsp_{}-{}_{}.txt".format(min_node_count, max_node_count, dataset_size)
    file = open(name, "a")
    file.write(str(dataset_size))
    file.write("\n\n")

    for g in range(dataset_size):
        node_count = random.randint(min_node_count, max_node_count)
        file.write(str(node_count))
        file.write("\n")
        graph = np.random.random([node_count, node_count]) * inverse_identity(node_count)
        for i in range(node_count):
            row = ""
            for j in range(node_count):
                row += str(graph[i, j])
                row += " "
            file.write(row)
            file.write("\n")


        # to find the shortest path, brute force is used
        best = np.inf
        bestpath = []
        allperms = list(itertools.permutations(list(range(1, node_count))))

        for perm in allperms:
            path = 0
            last = 0
            for item in perm:
                path += graph[last][item]
                last = item
            path += graph[last][0]
            if path < best:
                best = path
                bestpath = perm
        bestpath = list(bestpath)
        bestpath.insert(0, 0)

        row = ""
        for item in bestpath:
            row += str(item)
            row += " "
        file.write(row)

        file.write("\n\n")
        if g % 1000 == 0:
            print('Asymmetric TSP dataset: {}/{} done.'.format(g, dataset_size))

    file.close()
    return
