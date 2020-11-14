import tensorflow as tf


def sigmoid_log_loss(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor, eps=1e-8):
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
    return -(tf.math.log(1 - varsum + eps) - tf.math.log(1 + eps))


def softplus_log_square_loss(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor, eps=1e-8):
    return tf.square(softplus_log_loss(variable_predictions, clauses, eps))


def softplus_log_loss(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor, eps=1e-8):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :param eps: small value to avoid log(0)
    :return: returns per clause loss
    """
    clauses_val = softplus_loss(variable_predictions, clauses)

    return -(tf.math.log(1 - clauses_val + eps) - tf.math.log(1 + eps))


def softplus_loss(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :return: returns per clause loss in range [0..1] - 0 if clause is satisfied, 1 if not satisfied
    """
    clauses_split = clauses.row_lengths()
    flat_clauses = clauses.flat_values
    clauses_mask = tf.repeat(tf.range(0, clauses.nrows()), clauses_split)

    clauses_index = tf.abs(flat_clauses) - 1  # Just star indexing from 0. DIMACS standard start variables from 1
    variables = tf.gather(variable_predictions, clauses_index)  # Gather clauses of variables
    float_clauses = tf.cast(flat_clauses, tf.float32)
    variables = variables * tf.expand_dims(tf.sign(float_clauses), axis=-1)

    variables = tf.nn.softplus(variables)
    clauses_val = tf.math.segment_sum(variables, clauses_mask)
    clauses_val = tf.exp(-clauses_val)

    return clauses_val


def min_max_loss_per_clause(variable_prediction: tf.Tensor, clauses: tf.RaggedTensor, temp=1):
    """Implementation of softmin and softmax loss proposed in
    "Learning To Solve Circuit-SAT: An Unsupervised Differentiable Approach"

    Returns per clause loss. Suitable for in model loss.
    """
    variables = tf.sigmoid(variable_prediction)

    clauses_index = tf.abs(clauses) - 1  # Just star indexing from 0. DIMACS standard start variables from 1
    vars = tf.gather(variables, clauses_index)  # Gather clauses of variables

    # Inverse in form x for positive clause variable and (1-x) for negative representation
    float_clauses = tf.cast(clauses, tf.float32)
    variables = vars * tf.expand_dims(tf.sign(float_clauses), axis=-1)
    variables = tf.expand_dims(tf.clip_by_value(-float_clauses, 0, 1), axis=-1) + variables

    vars_with_temp = variables / temp
    exp = tf.exp(vars_with_temp)
    max_per_clause = tf.reduce_sum(exp * vars_with_temp, axis=-2) / tf.reduce_sum(exp, axis=-2)

    return max_per_clause


def min_max_loss(variable_prediction: tf.Tensor, clauses: tf.RaggedTensor, temp=1):
    """Implementation of softmin and softmax loss proposed in
    "Learning To Solve Circuit-SAT: An Unsupervised Differentiable Approach"

    Reduces loss as min over all clauses. Suitable for output. At the end applies step function loss propesed in paper.
    """
    clauses = min_max_loss_per_clause(variable_prediction, clauses, temp)
    clauses = tf.reduce_mean(clauses, axis=-1)

    # Reduce total
    exp = tf.exp(-clauses / temp)
    min_value = tf.reduce_sum(exp * clauses) / tf.reduce_sum(exp)

    skm = tf.pow(1 - min_value, 10)
    return skm / (skm + tf.pow(min_value, 10))