import tensorflow as tf


def variables_mul_loss(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor, eps=1e-8):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :param eps: small value to avoid log(0)
    :return: returns per clause loss
    """
    clauses_split = clauses.row_lengths()
    flat_clauses = clauses.flat_values
    clauses_mask = tf.repeat(tf.range(0, clauses.nrows()), clauses_split)

    variables = tf.sigmoid(variable_predictions)

    clauses_index = tf.abs(flat_clauses) - 1  # Just star indexing from 0. DIMACS standard start variables from 1
    vars = tf.gather(variables, clauses_index)  # Gather clauses of variables

    # Inverse in form (1 - x) for positive clause literal and x for negative
    float_clauses = tf.cast(flat_clauses, tf.float32)
    vars = vars * tf.expand_dims(tf.sign(float_clauses), axis=-1)
    inverse_vars = tf.expand_dims(tf.clip_by_value(float_clauses, 0, 1), axis=-1) - vars

    # multiply all values in a clause together
    varsum = tf.math.unsorted_segment_prod(inverse_vars, clauses_mask, clauses.nrows())
    return -tf.math.log(1 - varsum + eps)
