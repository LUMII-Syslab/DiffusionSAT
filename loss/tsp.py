import tensorflow as tf
from loss.unsupervised_tsp import tsp_unsupervised_loss
from loss.supervised_tsp import tsp_supervised_loss
from data.tsp import get_unpadded_size, remove_padding


def tsp_loss(predictions, adjacency_matrix, labels=None, noise=0, log_in_tb=False, fast_inaccurate=False, subtour_projection=False):
    """
    Cuts the padding of each batch element and passes them one by one to the loss functions as if each of
    the elements were a batch of size 1. Returns the total loss of the batch.
    :param predictions: tf.Tensor (batch_size, padded_size, padded_size)
    :param adjacency_matrix: tf.Tensor (batch_size, padded_size, padded_size)
    :param labels: tf.Tensor (batch_size, padded_size, padded_size) or None
    :param noise: int
    :param log_in_tb: bool
    :param fast_inaccurate: bool
    :param subtour_projection: bool
    :return: tf.Tensor ()
    """
    calculate_supervised = True  # True; False
    calculate_unsupervised = False  # True; False
    if labels is None: calculate_supervised = False

    batch_size, padded_size, *_ = tf.shape(predictions)
    predictions = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size])
    adjacency_matrix = tf.reshape(adjacency_matrix, shape=[batch_size, padded_size, padded_size])
    if labels is not None: labels = tf.reshape(labels, shape=[batch_size, padded_size, padded_size])
    loss = 0
    for i in range(batch_size):
        node_count = get_unpadded_size(adjacency_matrix[i])
        prediction = remove_padding(predictions[i], unpadded_size=node_count)
        graph = remove_padding(adjacency_matrix[i], unpadded_size=node_count)
        if labels is not None: label = remove_padding(labels[i], unpadded_size=node_count)
        prediction = tf.reshape(prediction, shape=[1, node_count, node_count])
        graph = tf.reshape(graph, shape=[1, node_count, node_count])
        if labels is not None: label = tf.reshape(label, shape=[1, node_count, node_count])

        if calculate_supervised:
            loss += 1*tsp_supervised_loss(prediction, label)
        if calculate_unsupervised:
            loss += 1*tsp_unsupervised_loss(prediction, graph, noise, log_in_tb, fast_inaccurate, subtour_projection)
    return loss