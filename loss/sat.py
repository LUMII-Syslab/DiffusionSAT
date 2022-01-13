import unittest

import tensorflow as tf


def unsat_clause_count(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :return: returns the number of not satisfied clauses. Calculated using rounded variables.
    """
    clauses_split = clauses.row_lengths()
    flat_clauses = clauses.flat_values
    clauses_mask = tf.repeat(tf.range(0, clauses.nrows()), clauses_split)

    variables = tf.round(tf.sigmoid(variable_predictions))

    clauses_index = tf.abs(flat_clauses) - 1  # Just star indexing from 0. DIMACS standard start variables from 1
    vars = tf.gather(variables, clauses_index)  # Gather clauses of variables

    # Inverse in form (1 - x) for positive clause literal and x for negative
    float_clauses = tf.cast(flat_clauses, tf.float32)
    vars = vars * tf.expand_dims(tf.sign(float_clauses), axis=-1)
    inverse_vars = tf.expand_dims(tf.clip_by_value(float_clauses, 0, 1), axis=-1) - vars

    # multiply all values in a clause together, satisfied clause is of value 0
    varsum = tf.math.unsorted_segment_prod(inverse_vars, clauses_mask, clauses.nrows())
    return tf.math.reduce_sum(varsum)  # count not satisfied ones


def unsat_cnf_clauses_loss(var_predictions: tf.Tensor, clauses_lit_adj: tf.SparseTensor, clauses_mask: tf.Tensor, eps=1e-5):
    clauses_val = softplus_loss(var_predictions, clauses_lit_adj, clauses_mask)
    return 1 - clauses_val


def unsat_cnf_loss(var_predictions: tf.Tensor, clauses_lit_adj: tf.SparseTensor, graph_clauses: tf.SparseTensor, clauses_mask: tf.Tensor, eps=1e-5):
    clauses_val = unsat_cnf_clauses_loss(var_predictions, clauses_lit_adj, clauses_mask, eps=eps)
    clauses_val = tf.math.log(clauses_val + eps) - tf.math.log1p(eps)

    per_graph_value = tf.sparse.sparse_dense_matmul(graph_clauses, clauses_val)
    clauses_count = tf.sparse.reduce_sum(graph_clauses * tf.squeeze(clauses_mask, axis=-1), axis=1)
    total_count = tf.sparse.reduce_sum(graph_clauses, axis=1)
    amortized_clauses = tf.sqrt(clauses_count + 2) - tf.sqrt(total_count)  # CNF UNSAT core should have at least two clauses
    amortized_clauses = -amortized_clauses  # / total_count

    return per_graph_value * amortized_clauses


def softplus_mixed_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor, clauses_mask: tf.Tensor, eps=1e-8):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :return: returns per clause loss which is log_loss multiplied with linear loss
    """
    clauses_val = softplus_loss(variable_predictions, adj_matrix, clauses_mask)
    log_clauses = -(tf.math.log(1 - clauses_val + eps) - tf.math.log1p(eps))
    return log_clauses


def softplus_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor, clauses_mask: tf.Tensor):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param adj_matrix: Sparse tensor with dense shape literals x clauses
    :return: returns per clause loss in range [0..1] - 0 if clause is satisfied, 1 if not satisfied
    """

    literals = tf.concat([variable_predictions, -variable_predictions], axis=0)
    literals = tf.nn.softplus(literals)
    clauses_val2 = tf.sparse.sparse_dense_matmul(adj_matrix, literals)  # Empty clauses should be handled as satisfied
    clauses_val = tf.exp(-clauses_val2) * clauses_mask

    return clauses_val


def linear_loss_adj(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param adj_matrix: Sparse tensor with dense shape literals x clauses
    :return: returns per clause loss in range [0..] - 0 if clause is satisfied, >0 if not satisfied
    """
    variable_predictions = tf.sigmoid(variable_predictions)
    literals = tf.concat([variable_predictions, 1 - variable_predictions], axis=0)
    clauses_val = tf.sparse.sparse_dense_matmul(adj_matrix, literals)

    zero_loss = tf.square(variable_predictions)
    ones_loss = tf.square(variable_predictions - 1)

    literal_loss = tf.reduce_sum(zero_loss * ones_loss)
    clauses_val = tf.nn.relu(1 - clauses_val)

    return tf.reduce_sum(clauses_val) + literal_loss


class TestSoftplusLoss(unittest.TestCase):

    def test_sat_instance(self):
        adj_matrix = tf.SparseTensor([[2, 0], [2, 1], [1, 1], [0, 2], [3, 2]], [1., 1., 1., 1., 1.], [4, 3])
        logits = tf.expand_dims(tf.constant([-20., -20.]), axis=-1)
        mask = tf.constant([[1.], [1.], [1.]])
        rez = softplus_loss(logits, tf.sparse.transpose(adj_matrix), mask)
        rez = rez.numpy().flatten()
        self.assertAlmostEqual(rez[0], 0.)
        self.assertAlmostEqual(rez[1], 0.)
        self.assertAlmostEqual(rez[2], 0.)

    def test_unsat_instance(self):
        adj_matrix = tf.SparseTensor([[2, 0], [0, 1], [1, 1], [0, 2], [3, 2]], [1., 1., 1., 1., 1.], [4, 3])
        logits = tf.expand_dims(tf.constant([-20., 20.]), axis=-1)
        mask = tf.constant([[1.], [1.], [1.]])
        rez = softplus_loss(logits, tf.sparse.transpose(adj_matrix), mask)
        rez = rez.numpy().flatten()
        self.assertAlmostEqual(rez[0], 0.)
        self.assertAlmostEqual(rez[1], 0.)
        self.assertAlmostEqual(rez[2], 1.)

    def test_sat_instance_mask(self):
        adj_matrix = tf.SparseTensor([[2, 0], [2, 1], [1, 1], [0, 2], [3, 2]], [1., 1., 1., 1., 1.], [4, 3])
        logits = tf.expand_dims(tf.constant([-20., -20.]), axis=-1)
        mask = tf.constant([[1.], [1.], [0.]])
        rez = softplus_loss(logits, tf.sparse.transpose(adj_matrix), mask)
        rez = rez.numpy().flatten()
        self.assertAlmostEqual(rez[0], 0.)
        self.assertAlmostEqual(rez[1], 0.)
        self.assertAlmostEqual(rez[2], 0.)

    def test_sat_instance_mask_05(self):
        adj_matrix = tf.SparseTensor([[2, 0], [2, 1], [1, 1], [0, 2], [3, 2]], [1., 1., 1., 1., 1.], [4, 3])
        logits = tf.expand_dims(tf.constant([-20., -20.]), axis=-1)
        mask = tf.constant([[1.], [1.], [0.5]])
        rez = softplus_loss(logits, tf.sparse.transpose(adj_matrix), mask)
        rez = rez.numpy().flatten()
        self.assertAlmostEqual(rez[0], 0.)
        self.assertAlmostEqual(rez[1], 0.)
        self.assertAlmostEqual(rez[2], 0.)


if __name__ == '__main__':
    adj_matrix = tf.SparseTensor([[2, 0], [0, 1], [1, 1], [0, 2], [3, 2]], [1., 1., 1., 1., 1.], [4, 3])
    logits = tf.constant([20., -20.])
    logits = tf.expand_dims(logits, axis=-1)

    graph_constr = tf.SparseTensor([[0, 0], [0, 1], [0, 2]], [1., 1., 1.], [1, 3])
    mask = tf.constant([[0.], [1.], [1.]])
    rez = unsat_cnf_loss(logits, tf.sparse.transpose(adj_matrix), graph_constr, mask)
    print(rez)
    # rez = unsat_cnf_clauses_loss(logits, tf.sparse.transpose(adj_matrix))
    # print(rez)

    rez = softplus_loss(logits, tf.sparse.transpose(adj_matrix), mask)
    print(1 - rez)
