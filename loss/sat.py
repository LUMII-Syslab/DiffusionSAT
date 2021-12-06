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


def unsat_cnf_clauses_loss(var_predictions: tf.Tensor, clauses_lit_adj: tf.SparseTensor, eps=1e-5):
    clauses_val = softplus_loss(var_predictions, clauses_lit_adj)
    return 1 - clauses_val


def unsat_cnf_loss(var_predictions: tf.Tensor, clauses_lit_adj: tf.SparseTensor, graph_clauses: tf.SparseTensor, eps=1e-5):
    clauses_val = unsat_cnf_clauses_loss(var_predictions, clauses_lit_adj, eps=eps)
    clauses_val = tf.square(clauses_val)
    clauses_val = tf.math.log(clauses_val + eps) - tf.math.log1p(eps)
    per_graph_value = tf.sparse.sparse_dense_matmul(graph_clauses, clauses_val)
    return per_graph_value


def softplus_mixed_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor, eps=1e-8):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :return: returns per clause loss which is log_loss multiplied with linear loss
    """
    clauses_val = softplus_loss(variable_predictions, adj_matrix)
    log_clauses = -(tf.math.log(1 - clauses_val + eps) - tf.math.log1p(eps))
    return clauses_val * log_clauses


def softplus_loss(variable_predictions: tf.Tensor, adj_matrix: tf.SparseTensor, power=1.):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param adj_matrix: Sparse tensor with dense shape literals x clauses
    :return: returns per clause loss in range [0..1] - 0 if clause is satisfied, 1 if not satisfied
    """

    literals = tf.concat([variable_predictions, -variable_predictions], axis=0)
    literals = tf.nn.softplus(literals)
    clauses_val2 = tf.sparse.sparse_dense_matmul(adj_matrix, literals)  # Empty clauses should be handled as satisfied

    not_empty_clause = tf.sparse.reduce_max(adj_matrix, axis=1, keepdims=True)
    not_empty_clause = tf.stop_gradient(not_empty_clause)  # Avoids tensorflow warnings
    clauses_val = tf.exp(-clauses_val2 * power) * not_empty_clause
    clauses_val = tf.ensure_shape(clauses_val, clauses_val2.shape)

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


if __name__ == '__main__':
    adj_matrix = tf.SparseTensor([[2, 0], [2, 1], [1, 1], [0, 2], [3, 2]], [1., 1., 1., 1., 1.], [4, 4])
    logits = tf.constant([-20., -20.])
    logits = tf.expand_dims(logits, axis=-1)

    graph_constr = tf.SparseTensor([[0, 0], [0, 1], [0, 2]], [1., 1., 1.], [1, 4])
    rez = unsat_cnf_loss(logits, tf.sparse.transpose(adj_matrix), graph_constr)
    print(rez)
    rez = unsat_cnf_clauses_loss(logits, tf.sparse.transpose(adj_matrix))
    print(rez)

    # rez = softplus_loss(logits, tf.sparse.transpose(adj_matrix))
    # print(rez)
