import tensorflow as tf
from loss.tsp_ineq import ineq


def sample_logistic(shape, eps=1e-20):
    U = tf.random.uniform(shape, minval=eps, maxval=1 - eps)
    return tf.math.log(U / (1 - U))


def inverse_identity(size):
    return tf.ones(shape=[size, size]) - tf.eye(size)


def tsp_unsupervised_loss(predictions, adjacency_matrix, noise=0):
    """
    :param predictions: TODO: Describe what and with what dimensions is expected as input
    :param adjacency_matrix:
    :param noise:
    :return:
    """

    # TODO(@Elīza): rename with meaningful variable names
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

    inequalities = ineq(x)
    x = tf.reshape(x, (batch_size, node_count * node_count, 1))

    cost4 = 0
    for inequality in inequalities:
        tmp = tf.sparse.sparse_dense_matmul(inequality[1], x[inequality[0]])
        cost4 += tf.reduce_sum(tf.pow(2 - tmp, 2)) / tf.cast(batch_size, tf.float32)

    cost4 *= 0.05
    # print(cost1.numpy()*5, cost2.numpy(), cost3.numpy(), cost4.numpy())
    return cost1 + cost2 + cost3 + cost4
