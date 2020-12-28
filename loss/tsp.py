import tensorflow as tf

from loss.unsupervised_tsp import tsp_unsupervised_loss, inverse_identity
from metrics.tsp_metrics import PADDING_VALUE


def tsp_loss(predictions, adjacency_matrix, labels=None, noise=0, log_in_tb=False, fast_inaccurate=False,
             subtour_projection=False, supervised=False, unsupervised=True):
    """
    Calculates supervised and unsupervised TSP loss of a batch.
    :param predictions: tf.Tensor (batch_size, padded_size, padded_size, 1)
    :param adjacency_matrix: tf.Tensor (batch_size, padded_size, padded_size)
    :param labels: tf.Tensor (batch_size, padded_size, padded_size) or None
    :param noise: int
    :param log_in_tb: bool
    :param fast_inaccurate: bool
    :param subtour_projection: bool
    :param unsupervised: bool
    :param supervised: bool
    :return: tf.Tensor ()
    """

    loss = 0
    if supervised:
        batch_size, padded_size, *_ = tf.shape(predictions)
        predictions = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size])
        mask = tf.cast(tf.not_equal(labels, PADDING_VALUE), tf.float32) * inverse_identity(padded_size)
        loss_tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels, predictions)
        loss_tensor = loss_tensor * mask  # sets padding and diagonal to zeros
        loss += tf.reduce_mean(loss_tensor)
    if unsupervised:
        loss += tsp_unsupervised_loss(predictions, adjacency_matrix, noise, log_in_tb, fast_inaccurate, subtour_projection)
    return loss
