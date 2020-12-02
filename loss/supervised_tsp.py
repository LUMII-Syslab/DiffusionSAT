import tensorflow as tf
from loss.unsupervised_tsp import inverse_identity


def tsp_supervised_loss(prediction, label):
    """
    Takes prediction and label tensors without padding and returns their mean squared error.
    :param prediction: tf.Tensor (1, node_count, node_count)
    :param label: tf.Tensor (1, node_count, node_count)
    :return: tfTensor ()
    """

    _, node_count, *_ = tf.shape(prediction)
    prediction = tf.sigmoid(prediction) * inverse_identity(node_count)  # gets sigmoid and sets the diagonal to zeros
    loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(label, prediction))
    return loss
