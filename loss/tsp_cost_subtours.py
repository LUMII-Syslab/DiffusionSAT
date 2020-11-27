import copy
import tensorflow as tf
import pyximport; pyximport.install()
from loss.subtours_cy import subtours

def subtour_constraints(predictions_tensor):
    batch_size, node_count, *_ = tf.shape(predictions_tensor)
    preds = predictions_tensor.numpy()
    predictions = list(copy.deepcopy(preds))

    subtours_cy = subtours(batch_size, node_count, predictions)

    subtour_py = []
    for g, subtour_edges, subtours_added in subtours_cy:
        subtour = tf.SparseTensor(values=[1.] * len(subtour_edges), indices=subtour_edges,
                                dense_shape=[subtours_added, node_count * node_count])
        subtour_py.append((g, subtour))

    return subtour_py
