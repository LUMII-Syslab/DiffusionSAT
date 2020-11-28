import os
import tensorflow as tf
import numpy as np

from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde


def inverse_identity(size):
    return tf.ones(shape=[size, size]) - tf.eye(size)


def tsp_supervised_loss(predictions, labels):
    """
    :param predictions: TODO: Describe what and with what dimensions is expected as input
    :param labels: matrices with optimal edges marked with ones
    :return: loss scalar
    """

    batch_size, node_count, *_ = tf.shape(predictions)

    predictions = tf.reshape(predictions, shape=[batch_size, node_count, node_count])
    predictions = tf.sigmoid(predictions) * inverse_identity(node_count)
    labels = tf.reshape(labels, shape=[batch_size, node_count, node_count])

    loss = 0
    for i in range(batch_size):  # iterate over the batch
        y_true = labels[i]
        y_pred = predictions[i]
        loss += tf.reduce_sum(tf.keras.losses.mean_squared_error(y_true, y_pred))
    return loss


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function. This will not suppress raised exceptions.
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''
    def __init__(self):
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]
    def __enter__(self):
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)
    def __exit__(self, *_):
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def get_path_with_Concorde(coordinates):
    with suppress_stdout_stderr():  # suppress_stdout_stderr prevents Concorde from printing
        # passing the x and y coordinates to the Concorde solver to find optimal Euclidean 2D TSP:
        solver = TSPSolver.from_data(coordinates[:, 0], coordinates[:, 1], norm="GEO")
        solution = solver.solve()
    # Only write instances with valid solutions
    assert (np.sort(solution.tour) == np.arange(len(coordinates))).all()  # all nodes are in the solution
    path = np.append(solution.tour, [solution.tour[0]])  # return to the first node
    return path

def get_score_with_Concorde(coordinates):
    # outputs a matrix with edges in the optimal path marked by ones.
    node_count = len(coordinates)
    path = get_path_with_Concorde(coordinates)
    output = np.zeros((node_count, node_count))
    index = path[0]
    for i in range(node_count):
        previous_index = index
        index = path[i+1]
        output[previous_index, index] = 1
        output[index, previous_index] = 1
    return output
