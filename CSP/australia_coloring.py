import tensorflow as tf

init = tf.keras.initializers.RandomNormal()

d = 3
x = tf.Variable(init([7, d], dtype=tf.float32))

if __name__ == '__main__':
    opt = tf.keras.optimizers.Adam(learning_rate=1)

    for i in range(1000):
        with tf.GradientTape() as tape:
            a = [5, 5, 5, 5, 5, 0, 1, 2, 3]
            b = [0, 1, 2, 3, 4, 1, 2, 3, 4]
            a_v = tf.gather(x, a, axis=0)
            b_v = tf.gather(x, b, axis=0)
            loss = -tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(a_v, tf.nn.softmax(b_v, axis=-1))
            )  # Inequality constraint
            print(loss)
            print(tf.argmax(x, axis=-1))

        grad = tape.gradient(loss, [x])
        opt.apply_gradients(zip(grad, [x]))
