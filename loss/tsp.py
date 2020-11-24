from loss.unsupervised_tsp import tsp_unsupervised_loss
from loss.supervised_tsp import tsp_supervised_loss

def tsp_loss(predictions, adjacency_matrix, coords, noise=0):
    supervised_loss = tsp_supervised_loss(predictions, adjacency_matrix, coords)
    unsupervised_loss = tsp_unsupervised_loss(predictions, adjacency_matrix, noise)
    return supervised_loss + unsupervised_loss