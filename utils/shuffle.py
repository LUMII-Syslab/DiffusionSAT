import math

import tensorflow as tf


def matrix_to_vector(matrix):
    """
    Converts matrix to vector according to Z-order-curve readout.
    """
    w = len(matrix)
    h = len(matrix[0])

    assert w == h, "Matrix dimensions should be equal"
    assert math.log(w * h, 4).is_integer(), "Total matrix element count should be power of 4"

    return __matrix_to_vector(matrix, w, 0, 0)


def __matrix_to_vector(matrix, length, x, y):
    if length == 1:
        return [matrix[x][y]]

    mid = length // 2

    res = []
    res += __matrix_to_vector(matrix, mid, x, y)
    res += __matrix_to_vector(matrix, mid, x, y + mid)
    res += __matrix_to_vector(matrix, mid, x + mid, y)
    res += __matrix_to_vector(matrix, mid, x + mid, y + mid)

    return res


def vector_to_matrix(vector):
    """
    Converts vector to matrix according to Z-order-curve readout.
    """
    length = len(vector)
    assert math.log(length, 4).is_integer(), "Total vector element count should be power of 4"
    return __vector_to_matrix(vector, 0, length)


def __vector_to_matrix(vector, start_pos, length):
    if length == 4:
        mid = start_pos + 2
        return [vector[start_pos:mid], vector[mid:start_pos + 4]]

    new_length = length // 4

    pos = [i for i in range(start_pos, start_pos + length, new_length)]

    first = __vector_to_matrix(vector, pos[0], new_length)
    second = __vector_to_matrix(vector, pos[1], new_length)
    third = __vector_to_matrix(vector, pos[2], new_length)
    fourth = __vector_to_matrix(vector, pos[3], new_length)

    res = []
    res += [a + b for a, b in zip(first, second)]
    res += [a + b for a, b in zip(third, fourth)]
    return res


def qrol(number, q_digits, stopped_digits=0):
    """Implement cyclic left shift for quaternary numbers with stopped_positions in right side"""
    return __quaternary_shift(rol, number, stopped_digits, q_digits)


def qror(number, q_digits, stopped_digits=0):
    """Implement cyclic right shift for quaternary numbers with stopped_positions in right side"""
    return __quaternary_shift(ror, number, stopped_digits, q_digits)


def __quaternary_shift(shift_operation, number, stopped_pos, q_digits):
    """
    :param shift_operation: ror or rol function
    :param number: input number
    :param stopped_pos: How many positions leave unchanged from the right side
    :return: shifted number
    """
    bits = q_digits * 2
    stopped_bits = stopped_pos * 2

    shifted_bits = shift_operation(number >> stopped_bits, bits - stopped_bits, 2)
    unchanged_bits = number & mask(stopped_bits)
    return (shifted_bits << stopped_bits) + unchanged_bits


def quaternary_digits(number) -> int:
    bits = number.bit_length()
    bits += 1 if bits % 2 == 1 else 0

    return bits // 2


def mask(bits):
    """Generate mask of 1's for n bits"""
    return 2 ** bits - 1


def ror(x, n, p=1):
    """Bitwise rotation right"""
    return (x >> p) + ((x & ((1 << p) - 1)) << (n - p))


def rol(x, n, p=1):
    """Bitwise rotation left"""
    return ((x << p) & ((1 << n) - 1)) | (x >> (n - p))


def gelu(x):
    """Implements Gaussian Error Linear Unit (GELU)"""
    return x * tf.sigmoid(1.702 * x)
