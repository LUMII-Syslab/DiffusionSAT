import tensorflow as tf


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
             experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
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
    return -tf.math.log(1 - varsum + eps)


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
             experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
def softplus_log_loss(variable_predictions: tf.Tensor, clauses: tf.RaggedTensor, eps=1e-8):
    """
    :param variable_predictions: Logits (without sigmoid applied) from model output - each element represents variable
    :param clauses: RaggedTensor of input clauses in DIMAC format
    :param eps: small value to avoid log(0)
    :return: returns per clause loss
    """
    clauses_val = softplus_loss(variable_predictions, clauses)

    return -(tf.math.log(1 - clauses_val + eps)-tf.math.log(1+eps))

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32),
                              tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, row_splits_dtype=tf.int32)],
             experimental_autograph_options=tf.autograph.experimental.Feature.ALL)
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
