import tensorflow as tf
import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from loss.tsp_subtours_cy import subtours
from data.tsp import PADDING_VALUE



def tsp_unsupervised_loss(predictions, adjacency_matrix, noise=0, log_in_tb=False, fast_inaccurate=False, subtour_projection=False):
    """
    :param predictions: TODO: Describe what and with what dimensions is expected as input
    :param adjacency_matrix:
    :param noise:
    :return:
    """

    batch_size, padded_size, *_ = tf.shape(predictions)
    adjacency_matrix = tf.reshape(adjacency_matrix, shape=[batch_size, padded_size, padded_size])
    mask = tf.cast(tf.not_equal(adjacency_matrix, PADDING_VALUE), tf.float32) * inverse_identity(padded_size)

    predictions = tf.reshape(predictions, shape=[batch_size, padded_size, padded_size])
    # predictions = predictions * mask


    # todo: maybe this method should use already post-sigmoid predictions
    distribution = sample_logistic(shape=[batch_size, padded_size, padded_size])
    predictions = predictions + distribution * noise
    predictions = tf.sigmoid(predictions) * inverse_identity(padded_size)

    cost_incoming = tf.reduce_mean(tf.square(1 - tf.reduce_sum(predictions * mask, 1)))
    cost_outgoing = tf.reduce_mean(tf.square(1 - tf.reduce_sum(predictions * mask, 2)))
    predictions = predictions / (tf.reduce_sum(predictions * mask, 1, keepdims=True) + 1e-6)
    predictions = predictions / (tf.reduce_sum(predictions * mask, 2, keepdims=True) + 1e-6)

    cost_subtours = 0
    if fast_inaccurate or False:  # todo šis ar padding
        sum_with_reverse = predictions + tf.transpose(predictions, [0, 2, 1])
        cost_subtours = tf.reduce_sum(tf.square(tf.nn.relu(sum_with_reverse - 1))) / tf.cast(batch_size, tf.float32)

    else:
        subtours_cy = subtours(batch_size.numpy(), padded_size.numpy(), predictions.numpy(), adjacency_matrix.numpy(), PADDING_VALUE)

        predictions = tf.reshape(predictions, (batch_size * padded_size * padded_size, 1))

        if subtours_cy:
            subtours_sparse = tf.SparseTensor(values=[1.] * len(subtours_cy), indices=subtours_cy,
                                              dense_shape=[subtours_cy[-1][0] + 1, batch_size * padded_size * padded_size])
            cut_weight = tf.sparse.sparse_dense_matmul(subtours_sparse, predictions)  # All these cut_weight values are < 2
            cost_subtours += tf.reduce_sum(tf.square(1 - cut_weight)) / tf.cast(batch_size, tf.float32)

            if subtour_projection and False:  # todo šis ar padding
                # add constraint cut_weight >= 2.
                dif = (2 - cut_weight) / tf.sparse.reduce_sum(subtours_sparse, axis=1, keepdims=True)  # how much each prediction should be increased
                prediction_dif = tf.sparse.sparse_dense_matmul(subtours_sparse, dif, adjoint_a=True)  # convert to the space of predicions
                prediction_weight = tf.expand_dims(tf.sparse.reduce_sum(subtours_sparse, axis=0), axis=-1)  # when several subtours affect one edge, the average should be taken
                predictions = predictions + prediction_dif / tf.maximum(prediction_weight, 1.)

                # cut_weight1 = tf.sparse.sparse_dense_matmul(subtours_sparse, predictions)  # sanity check
                # print(cut_weight, cut_weight1)
                # print("")

        predictions = tf.reshape(predictions, [batch_size, padded_size, padded_size])

    # todo normalized ar padding
    adjacency_normalized = adjacency_matrix * mask * tf.math.rsqrt(tf.reduce_mean(tf.square(adjacency_matrix * mask), axis=[1, 2], keepdims=True) + 1e-6)
    cost_length = tf.reduce_mean(predictions * adjacency_normalized * mask)

    if log_in_tb:
        tf.summary.scalar("cost/length", cost_length)
        tf.summary.scalar("cost/incoming", cost_incoming)
        tf.summary.scalar("cost/outgoing", cost_outgoing)
        tf.summary.scalar("cost/subtours", cost_subtours)

    # scale to return values in reasonable range
    return cost_length * 5 + cost_incoming + cost_outgoing + cost_subtours * 0.05


def sample_logistic(shape, eps=1e-20):
    sample = tf.random.uniform(shape, minval=eps, maxval=1 - eps)
    return tf.math.log(sample / (1 - sample))


def inverse_identity(size):
    return tf.ones(shape=[size, size]) - tf.eye(size)
