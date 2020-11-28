from loss.unsupervised_tsp import tsp_unsupervised_loss
from loss.supervised_tsp import tsp_supervised_loss

def tsp_loss(predictions, adjacency_matrix, labels, noise=0):
    # hybrid loss
    # supervised_loss = tsp_supervised_loss(predictions, labels)
    # unsupervised_loss = tsp_unsupervised_loss(predictions, adjacency_matrix, noise, fast_inaccurate=True)
    # return 1.0*supervised_loss + 1.0*unsupervised_loss

    # just unsupervised loss (faster than hybrid loss):
    # unsupervised_loss = tsp_unsupervised_loss(predictions, adjacency_matrix, noise)
    # return unsupervised_loss

    # just supervised loss (faster than hybrid loss):
    supervised_loss = tsp_supervised_loss(predictions, labels)
    return supervised_loss
