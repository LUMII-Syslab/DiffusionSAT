import tensorflow as tf
from loss.tsp_cost_subtours import subtour_constraints

def sample_logistic(shape, eps=1e-20):
    sample = tf.random.uniform(shape, minval=eps, maxval=1 - eps)
    return tf.math.log(sample / (1 - sample))


def inverse_identity(size):
    return tf.ones(shape=[size, size]) - tf.eye(size)


def tsp_unsupervised_loss(predictions, adjacency_matrix, noise=0):
    """
    :param predictions: TODO: Describe what and with what dimensions is expected as input
    :param adjacency_matrix:
    :param noise:
    :return:
    """

    batch_size, node_count, *_ = tf.shape(predictions)
    graph = tf.reshape(adjacency_matrix, shape=[batch_size, node_count, node_count])

    distribution = sample_logistic(shape=[batch_size, node_count, node_count])
    predictions = tf.reshape(predictions, shape=[batch_size, node_count, node_count]) + distribution * noise
    predictions = tf.sigmoid(predictions) * inverse_identity(node_count)

    cost_incoming = tf.reduce_mean(tf.square(1 - tf.reduce_sum(predictions, 1)))
    cost_outgoing = tf.reduce_mean(tf.square(1 - tf.reduce_sum(predictions, 2)))
    predictions = predictions / (tf.reduce_sum(predictions, 1, keepdims=True) + 1e-6)
    predictions = predictions / (tf.reduce_sum(predictions, 2, keepdims=True) + 1e-6)
    cost_length = tf.reduce_mean(predictions * graph)

    cost_subtours = 0
    subtours = subtour_constraints(predictions)
    predictions = tf.reshape(predictions, (batch_size, node_count * node_count, 1))

    for i, subtour in subtours:
        tmp = tf.sparse.sparse_dense_matmul(subtour, predictions[i])
        cost_subtours += tf.reduce_sum(tf.square(2 - tmp)) / tf.cast(batch_size, tf.float32)

    cost_subtours *= 0.05
    return cost_length + cost_incoming + cost_outgoing + cost_subtours
