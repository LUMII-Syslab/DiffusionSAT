import tensorflow as tf

def real_and(x,y):
    val = (1-x)*(1-y)/4
    return 1-2*val

def anf_value_real(logits: tf.Tensor, ands_index1:tf.Tensor,ands_index2:tf.Tensor,clauses_adj:tf.SparseTensor):
    """"
    ands_index1 mapping from vars to first and value of clauses
    ands_index2 mapping from vars to second and value of clauses
    clauses_index monotonic map from and_vals to clauses
    """
    n_maps = tf.shape(logits)[-1]
    one = -tf.ones([1,n_maps])  # zero is represented as+1, one as -1
    values = tf.tanh(logits)
    values = tf.concat([one, values], axis=0)
    ands1 = tf.gather(values, ands_index1, axis=0)
    ands2 = tf.gather(values, ands_index2, axis=0)
    and_val = real_and(ands1, ands2)
    values_ands = tf.concat([values, and_val], axis=0)
    log_val = tf.math.log(tf.abs(values_ands)+1e-16)
    signs = (1-tf.sign(values_ands))/2
    sum_logs = tf.sparse.sparse_dense_matmul(clauses_adj,log_val, adjoint_a=True)
    sum_signs = tf.sparse.sparse_dense_matmul(clauses_adj, signs, adjoint_a=True)
    #sum_signs = tf.cast(1-2*(tf.cast(sum_signs, tf.int32) % 2), tf.float32)
    sum_signs = 1-2*tf.math.floormod(sum_signs, 2)
    clause_value = tf.exp(sum_logs)*sum_signs
    return clause_value, ands1, ands2

def normalize(x):
    x_real, x_im = tf.split(x, 2, axis=-1)
    len = tf.math.rsqrt(tf.square(x_real)+tf.square(x_im)+1e-6)
    len = tf.minimum(len, 1.0)
    return tf.concat([x_real*len, x_im*len], axis=-1)

def cplx_and(a, b):
    a_real, a_im = tf.split(a, 2, axis=-1)
    b_real, b_im = tf.split(b, 2, axis=-1)
    a1_real = (1 - a_real) / 2
    b1_real = (1 - b_real) / 2
    a1_im = -a_im / 2
    b1_im = -b_im / 2
    re = a1_real*b1_real
    im = a1_real * b1_im + a1_im * b1_real
    return 1 - 2 * re, -2 * im


# def anf_value_cplx(logits: tf.Tensor, ands_index1:tf.Tensor,ands_index2:tf.Tensor,clauses_index:tf.Tensor):
#     """"
#     ands_index1 mapping from vars to first and value of clauses
#     ands_index2 mapping from vars to second and value of clauses
#     clauses_index monotonic map from and_vals to clauses
#     """
#     n_maps = tf.shape(logits)[-1]//2
#     one = tf.concat([-tf.ones([1,n_maps]), tf.zeros([1,n_maps])], axis=-1)  # zero is represented as +1+0j, one as -1+0j
#     #values = tf.tanh(logits)
#     values = normalize(logits)
#     values = tf.concat([one, values], axis=0)
#     ands1 = tf.gather(values, ands_index1, axis=0)
#     ands2 = tf.gather(values, ands_index2, axis=0)
#     and_real, and_im = cplx_and(ands1, ands2)
#     angle = tf.math.atan2(and_im, and_real)
#     log_len = 0.5*tf.math.log(tf.square(and_real)+tf.square(and_im)+1e-16)
#     sum_angles = tf.math.segment_sum(angle, clauses_index)
#     sum_len = tf.math.segment_sum(log_len, clauses_index)
#     clause_real = tf.exp(sum_len)*tf.math.cos(sum_angles)
#     clause_im = tf.exp(sum_len) * tf.math.sin(sum_angles)
#     return clause_real, clause_im

def anf_value_cplx_adj(logits: tf.Tensor, ands_index1:tf.Tensor,ands_index2:tf.Tensor,clauses_adj:tf.SparseTensor, use_norm=False):
    """"
    ands_index1 mapping from vars to first and value of clauses
    ands_index2 mapping from vars to second and value of clauses
    clauses_index monotonic map from and_vals to clauses
    """
    n_maps = tf.shape(logits)[-1]//2
    one = tf.concat([-tf.ones([1,n_maps]), tf.zeros([1,n_maps])], axis=-1)  # zero is represented as +1+0j, one as -1+0j
    values = tf.tanh(logits)
    if use_norm: values = normalize(values)
    values = tf.concat([one, values], axis=0)
    ands1 = tf.gather(values, ands_index1, axis=0)
    ands2 = tf.gather(values, ands_index2, axis=0)
    and_real0, and_im0 = cplx_and(ands1, ands2)
    val_real, val_im = tf.split(values, 2, axis=-1)
    and_real = tf.concat([val_real, and_real0], axis=0)
    and_im = tf.concat([val_im, and_im0], axis=0)
    angle = tf.math.atan2(and_im, and_real)
    log_len = 0.5*tf.math.log(tf.square(and_real)+tf.square(and_im)+1e-16)
    sum_angles = tf.sparse.sparse_dense_matmul(clauses_adj,angle, adjoint_a=True)
    sum_len = tf.sparse.sparse_dense_matmul(clauses_adj, log_len, adjoint_a=True)
    clause_real = tf.exp(sum_len)*tf.math.cos(sum_angles)
    clause_im = tf.exp(sum_len) * tf.math.sin(sum_angles)
    return clause_real, clause_im, ands1, ands2

def anf_loss_cplx(logits: tf.Tensor, ands_index1:tf.Tensor,ands_index2:tf.Tensor,clauses_index:tf.Tensor, n_clauses):
    clause_real, clause_im = anf_value_cplx(logits, ands_index1, ands_index2, clauses_index, n_clauses)
    return 1-clause_real

def anf_loss_cos(logits: tf.Tensor, ands_index1:tf.Tensor,ands_index2:tf.Tensor,clauses_index:tf.Tensor, n_clauses):
    """"
    ands_index1 mapping from vars to first and value of clauses
    ands_index2 mapping from vars to second and value of clauses
    clauses_index monotonic map from and_vals to clauses
    """
    one = -tf.ones_like(logits[0:1,:])  # zero is represented as+1, one as -1
    values = tf.tanh(logits)
    values = tf.concat([one, values], axis=0)
    ands1 = tf.gather(values, ands_index1, axis=0, validate_indices=True)
    ands2 = tf.gather(values, ands_index2, axis=0, validate_indices=True)
    and_val = real_and(ands1, ands2)
    # log_val = tf.math.log(tf.abs(and_val)+1e-6)
    # signs = (1-tf.cast(tf.sign(and_val), tf.int32))//2
    log_val = tf.math.acos(and_val*0.999)
    sum_logs = tf.math.segment_sum(log_val, clauses_index)
    #sum_signs = tf.math.segment_sum(signs, clauses_index)
    #sum_signs = tf.cast(1-2*(sum_signs % 2), tf.float32)
    clause_value = tf.cos(sum_logs)
    #clause_value = tf.exp(sum_logs)*sum_signs
    # if tf.math.is_nan(tf.reduce_sum(clause_value)):
    #     print("sum",tf.reduce_sum(clause_value))
    #
    # if tf.math.is_nan(tf.reduce_mean(clause_value)):
    #     print(clause_value)
    return clause_value


def anf_loss_xx(cplx_logits: tf.Tensor, adj_matrix1: tf.SparseTensor, adj_matrix2: tf.SparseTensor):

    #cplx_logits =
    real_part, im_part = tf.split(cplx_logits, 2, axis=-1)
    real_part = tf.concat([tf.ones_like(real_part[0,:]),real_part], axis=0)# add cplx-zero at index 0
    im_part = tf.concat([tf.zeros_like(im_part[0, :]), im_part], axis=0)  # add cplx-zero at index 0

    val1 = tf.gather(cplx_logits, adj_matrix1.indices[:, 0])
    val2 = tf.gather(cplx_logits, adj_matrix2.indices[:, 0])
    val = cplx_and(val1, val2)
    lit_val = tf.math.unsorted_segment_prod(val, adj_matrix.indices[:, 0], adj_matrix.dense_shape[0])


    clauses_val1 = tf.sparse.sparse_dense_matmul(adj_matrix, literals)
    clauses_val = tf.exp(-clauses_val * power)
    units = tf.concat([q, k], axis=-1)
    weights = tf.sigmoid(self.unit_mlp(units))
    # lit_val = tf.math.segment_sum(units, adj_matrix.indices[:,1]) # are indices sorted?
    lit_val = tf.math.unsorted_segment_sum(k * weights, adj_matrix.indices[:, 0], adj_matrix.dense_shape[0])

    return clauses_val

# @tf.custom_gradient
# def segment_prod(data:tf.Tensor, segment_ids:tf.Tensor, depth=0):
#     if depth>3:
#         result = tf.math.segment_prod(data, segment_ids)
#     else:
#        result = segment_prod(data, segment_ids, depth+1)
#     #result = tf.cond(depth>3,lambda: tf.math.segment_prod(data, segment_ids),lambda:segment_prod(data, segment_ids, depth+1))
#     def _SegmentProdGrad(grad):
#         """Gradient for SegmentProd.
#         The gradient can be expressed for each segment by dividing the segment's
#         product by each element of the segment input tensor, but this approach can't
#         deal with zeros in the input.
#         Unlike reduce_prod we can't use cumsum here as individual segments may have
#         a different number of elements. Therefore we consider three cases:
#         1) A segment input contains no zeros and we can safely divide by the input
#            tensor.
#         2) A segment contains exactly one zero. Then the gradient of each input of
#            the segment is zero except for the 0-input, there the gradient is
#            the product of the remaining segment entries.
#         3) A segment contains at least two zeros. The gradient is zero for all
#            segment inputs.
#         """
#         is_zero = tf.math.equal(data, 0)
#         num_zeros = tf.math.segment_sum(
#             tf.cast(is_zero, dtype=tf.int32), segment_ids)
#         # handle case 3 and set the gradient to 0 for segments with more than one
#         # 0 as input
#         grad = tf.where(
#             tf.math.greater(num_zeros, 1), tf.zeros_like(grad), grad)
#         # replace all zeros with ones and compute the segment_prod
#         non_zero_data = tf.where(is_zero, tf.ones_like(data), data)
#         non_zero_prod = segment_prod(non_zero_data, segment_ids,0)
#         gathered_prod = tf.gather(result, segment_ids)
#         gathered_non_zero_prod = tf.gather(non_zero_prod, segment_ids)
#         prod_divided_by_el = gathered_prod / non_zero_data
#         # Now fetch the individual results for segments containing 0 and those that
#         # don't.
#         partial_derivative = tf.where(is_zero, gathered_non_zero_prod,
#                                                 prod_divided_by_el)
#         gathered_grad = tf.gather(grad, segment_ids)
#         return gathered_grad * partial_derivative, None, None
#
#     return result, _SegmentProdGrad

from tensorflow.python.framework import function
from tensorflow.python.framework import ops

#@function.Defun(func_name="slog_grad")
@ops.RegisterGradient("SegmentProd")
def _SegmentProdGrad(op, grad):
    """Gradient for SegmentProd.
    The gradient can be expressed for each segment by dividing the segment's
    product by each element of the segment input tensor, but this approach can't
    deal with zeros in the input.
    Unlike reduce_prod we can't use cumsum here as individual segments may have
    a different number of elements. Therefore we consider three cases:
    1) A segment input contains no zeros and we can safely divide by the input
       tensor.
    2) A segment contains exactly one zero. Then the gradient of each input of
       the segment is zero except for the 0-input, there the gradient is
       the product of the remaining segment entries.
    3) A segment contains at least two zeros. The gradient is zero for all
       segment inputs.
    """
    data = op.inputs[0]
    segment_ids = op.inputs[1]
    result = op.outputs[0]
    is_zero = tf.math.equal(data, 0)
    num_zeros = tf.math.segment_sum(
        tf.cast(is_zero, dtype=tf.int32), segment_ids)
    # handle case 3 and set the gradient to 0 for segments with more than one
    # 0 as input
    grad = tf.where(
        tf.math.greater(num_zeros, 1), tf.zeros_like(grad), grad)
    # replace all zeros with ones and compute the segment_prod
    non_zero_data = tf.where(is_zero, tf.ones_like(data), data)
    non_zero_prod = segment_prod(non_zero_data, segment_ids)
    gathered_prod = tf.gather(result, segment_ids)
    gathered_non_zero_prod = tf.gather(non_zero_prod, segment_ids)
    prod_divided_by_el = gathered_prod / (tf.abs(non_zero_data)+1e-6)*tf.sign(non_zero_data)
    # Now fetch the individual results for segments containing 0 and those that
    # don't.
    partial_derivative = tf.where(is_zero, gathered_non_zero_prod,
                                  prod_divided_by_el)
    gathered_grad = tf.gather(grad, segment_ids)
    return gathered_grad * partial_derivative, None


# @function.Defun(func_name="slog_grad")
# def slog_grad(x, grad):
#     return grad / (tf.abs(x) + 1)


# def slog_shape(x_op):
#     return [x_op.inputs[0].get_shape()]


#@function.Defun(grad_func=_SegmentProdGrad)
def segment_prod(data, segment_ids):
    return tf.math.segment_prod(data, segment_ids)
