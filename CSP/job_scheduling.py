import tensorflow as tf

"""
X = {
0: Axle F,
1: Axle B,
2: Wheel RF,
3: Wheel LF,
4: Wheel RB,
5: Wheel LB,
6: Nuts RF,
7: Nuts LF,
8: Nuts RB,
9: Nuts LB,
10: Cap RF,
11: Cap LF,
12: Cap RB,
13: Cap LB,
14: Inspect
} .
"""

"""
Axle F + 10 ≤ Wheel RF ; Axle F + 10 ≤ Wheel LF ;
Axle B + 10 ≤ Wheel RB ; Axle B + 10 ≤ Wheel LB .
"""
init = tf.keras.initializers.RandomNormal()
bits = 5
var = tf.Variable(init([2], dtype=tf.float32))
binary_values = [2 ** k for k in range(0, bits)]


def sample_logistic(shape, eps=1e-20):
    sample = tf.random.uniform(shape, minval=eps, maxval=1 - eps)
    return tf.math.log(sample / (1 - sample))


@tf.custom_gradient
def diff_round(x):
    def grad(dy):
        return dy

    return tf.round(x), grad


if __name__ == '__main__':
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(10):
        for step in range(1000):
            with tf.GradientTape() as tape:
                rez = diff_round(var)
                x = rez[0]
                y = rez[1]

                loss_0 = tf.nn.relu(-x + y - 1)
                loss_1 = tf.nn.relu(3 * x + 2 * y - 12)
                loss_2 = tf.nn.relu(2 * x + 3 * y - 12)
                loss_3 = tf.nn.relu(-x)
                loss_4 = tf.nn.relu(-y)
                loss_max = -y

                # loss_l = tf.reduce_sum(tf.nn.relu(32 - rez2))
                # loss_u = tf.reduce_sum(tf.nn.relu(rez2))
                #
                # loss_0 = tf.nn.relu(5 - rez2[0])
                # loss_1 = tf.nn.relu(rez2[0] + 10 - rez2[1])
                # loss_2 = tf.nn.relu(rez2[1] - 25)
                # loss_3 = tf.nn.relu(20 - rez2[1])
                #
                # loss_4 = tf.nn.relu(rez2[2] - 31)
                # loss_5 = tf.nn.relu(31 - rez2[2])

                sys_loss = 0.8 * (loss_0 + loss_1 + loss_2 + loss_3 + loss_4) + 0.2 * loss_max

                grad = tape.gradient(sys_loss, [var])
                opt.apply_gradients(zip(grad, [var]))

        print("Step", i, "Loss0", loss_0.numpy(), "Loss1", loss_1.numpy(), "Loss2", loss_2.numpy(), "Loss3",
              loss_3.numpy(), "Total_loss", sys_loss.numpy())
        print(rez.numpy().tolist())
