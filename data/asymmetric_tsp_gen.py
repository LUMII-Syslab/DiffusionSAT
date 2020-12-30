import random
import numpy as np
import itertools

# Generate a file of asymmetric tsp graphs of COUNT of sizes from MIN to MAX with labels
COUNT = 100
MIN = 6
MAX = 8


name = "asymmetric_tsp_" + str(MIN) + "-" + str(MAX) + "_" + str(COUNT) + ".txt"
file = open(name, "a")
file.write(str(COUNT))
file.write("\n\n")


def inverse_identity(size):
    return np.ones(shape=[size, size]) - np.eye(size)


for g in range(COUNT):
    n = random.randint(MIN, MAX)
    file.write(str(n))
    file.write("\n")
    graph = np.random.random([n, n]) * inverse_identity(n)
    for i in range(n):
        row = ""
        for j in range(n):
            row += str(graph[i, j])
            row += " "
        file.write(row)
        file.write("\n")


    # to find the shortest path, brute force is used

    best = np.inf
    bestpath = []
    allperms = list(itertools.permutations(list(range(1, n))))

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

file.close()
