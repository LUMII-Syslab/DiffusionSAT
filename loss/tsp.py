import tensorflow as tf
from loss.unsupervised_tsp import tsp_unsupervised_loss, inverse_identity
from data.tsp import PADDING_VALUE


def tsp_loss(predictions, adjacency_matrix, labels=None, noise=0, log_in_tb=False, fast_inaccurate=False,
             subtour_projection=False, supervised=False, unsupervised=True):
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
    :param unsupervised: bool
    :param supervised: bool
    :return: tf.Tensor ()
    """
    assert(not(supervised and (labels is None)))

    loss = 0
    if supervised:
        batch_size, padded_size, *_ = tf.shape(predictions)
        predictions = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size])
        labels = tf.reshape(labels, shape=[batch_size, padded_size, padded_size])
        predictions = tf.sigmoid(predictions) * inverse_identity(padded_size)  # sigmoid; sets the diagonal to zeros
        padding_mask = tf.cast(tf.not_equal(labels, PADDING_VALUE), tf.float32)
        loss_tensor = tf.math.square(labels - predictions)
        loss += tf.reduce_mean(padding_mask * loss_tensor)
    if unsupervised:
        loss += tsp_unsupervised_loss(predictions, adjacency_matrix, noise, log_in_tb, fast_inaccurate, subtour_projection)
    return loss
