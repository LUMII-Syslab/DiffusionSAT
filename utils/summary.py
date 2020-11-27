import io
import tensorflow as tf
import matplotlib.pyplot as plt


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

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image