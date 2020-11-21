import copy
import tensorflow as tf


def ineq(predictions):
    ineq = []

    x2 = copy.deepcopy(predictions.numpy())  # TODO(@Elīza): I believe we can live without deepcopy and numpy if tenor is used
    # TODO(@Elīza): Optimize this, can something be cached or precomputed, can we parallelize it along batch dimension?
    # TODO(@Elīza): rename with meaningful variable names

    batch_size, node_count, *_ = tf.shape(predictions)

    for g in range(len(x2)):
        graph = x2[g]
        G = []  # G* - sakārto šķautnes pēc svariem
        for i in range(node_count):
            for j in range(node_count):
                G.append((graph[i][j] + graph[j][i], (i, j)))
        G.sort(reverse=True)

        ind = []
        cnt = 0
        components = list(range(node_count))
        for edge in G:  # šis notiks O(n) reizes
            i = edge[1][0]  # šķautnes virsotnes
            j = edge[1][1]

            if components[i] != components[j]:  # ja šķautne pieder dažādām F komponentēm
                components[:] = [components[i] if x == components[j] else x for x in components]
                S = {i, j}  # S kopai pievieno visas virsotnes no tās F komponentes, kurā ir edge

                for k in range(len(components)):
                    if k != i and k != j and components[k] == components[i]:
                        S.add(k)

                if len(S) == node_count:
                    break

                sum = 0
                ind_tmp = []
                for i in range(node_count):
                    for j in range(node_count):
                        if (i in S and j not in S) or (j in S and i not in S):
                            sum += graph[i][j]
                            ind_tmp.append([cnt, node_count * i + j])

                if sum < 2:
                    ind.extend(ind_tmp)
                    cnt += 1

        if cnt != 0:
            A = tf.SparseTensor(values=[1.] * len(ind), indices=ind,
                                dense_shape=[cnt, node_count * node_count])
            ineq.append((g, A))

    return ineq
