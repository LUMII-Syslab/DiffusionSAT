import tensorflow as tf


def log_as_histogram(name, values):
    """Logs values as histogram."""
    # todo write values directly into histogram
    values *= tf.math.rsqrt(tf.reduce_mean(tf.square(values)) + 1e-10)
    data = []
    scale = 100
    for i in range(values.shape[0]):
        d = tf.zeros(tf.cast(tf.round(scale * values[i]), dtype=tf.int64)) + i
        data.append(d)

    data = tf.concat(data, axis=0)
    data = tf.cast(data, tf.float32) + tf.random.uniform(tf.shape(data), -0.5, 0.5)

    tf.summary.histogram(name, data, buckets=values.shape[0] + 1)
