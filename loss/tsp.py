import copy

import tensorflow as tf
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph._traversal import connected_components


def sample_logistic(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=eps, maxval=1 - eps)
    return tf.math.log(U / (1 - U))


def inverse_identity(size):
    return tf.ones(shape=[size, size]) - tf.eye(size)


def tsp_loss(predictions, adjacency_matrix, noise=0):
    batch_size, node_count, *_ = tf.shape(predictions)
    u = sample_logistic(shape=[batch_size, node_count, node_count])
    graph = tf.reshape(adjacency_matrix, shape=[batch_size, node_count, node_count])

    x = tf.reshape(predictions, shape=[batch_size, node_count, node_count]) + u * noise
    x = tf.sigmoid(x) * inverse_identity(node_count)  # ietver 1. nosacījumu

    cost2 = tf.reduce_mean((1 - tf.reduce_sum(x, 1)) ** 2)  # 2. nosacījums
    cost3 = tf.reduce_mean((1 - tf.reduce_sum(x, 2)) ** 2)  # 3. nosacījums
    x = x / (tf.reduce_sum(x, 1, keepdims=True) + 1e-10)
    x = x / (tf.reduce_sum(x, 2, keepdims=True) + 1e-10)
    cost1 = tf.reduce_mean(x * graph)  # minimizējamais vienādojums

    AA = []
    x2 = copy.deepcopy(x.numpy())  # TODO: I believe we can live without deepcopy and numpy if tenor is used
    for g in range(len(x2)):
        graph = x2[g]
        G = []  # G* - sakārto šķautnes pēc svariem
        for i in range(node_count):
            for j in range(node_count):
                G.append((graph[i][j] + graph[j][i], (i, j)))
        G.sort(reverse=True)

        ind = []
        cnt = 0
        F = lil_matrix((node_count, node_count))
        for edge in G:  # šis notiks O(n) reizes
            i = edge[1][0]  # šķautnes virsotnes
            j = edge[1][1]

            C = connected_components(F)[1]  # F grafa komponentes
            if C[i] != C[j]:  # ja šķautne pieder dažādām F komponentēm
                F[i, j] = 1  # pievieno šķautni F grafam
                F[j, i] = 1  # abos virzienos?
                S = {i, j}  # S kopai pievieno visas virsotnes no tās F komponentes, kurā ir edge

                for k in range(len(C)):
                    if k != i and k != j and (C[k] == C[i] or C[k] == C[j]):
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

        if (cnt != 0):
            adj_matrix = tf.SparseTensor(values=[1.] * len(ind), indices=ind,
                                         dense_shape=[cnt, node_count * node_count])
            AA.append([g, adj_matrix])

    cost4 = 0.
    x = tf.reshape(x, (batch_size, node_count * node_count, 1))
    for a_tensor in AA:
        nr = a_tensor[0]
        adj_matrix = a_tensor[1]
        tmp = tf.sparse.sparse_dense_matmul(adj_matrix, x[nr])
        cost4 += tf.reduce_sum(2. - tmp) / tf.cast(batch_size, dtype=tf.float32)  # ņem summu visiem viena grafa pārkāpumiem

    # multiplied_dims = []
    # for i in range(x.shape[-1]):
    #     multiplied_dims.append(tf.expand_dims(tf.sparse.reduce_sum(A * x[:, :, i][:, None, :], axis=-1), -1))
    # result = tf.concat(multiplied_dims, -1)

    cost4 *= 0.05
    # print(cost1.numpy()*5, cost2.numpy(), cost3.numpy(), cost4.numpy())
    return cost1 + cost2 + cost3 + cost4
