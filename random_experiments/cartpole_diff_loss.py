import gym
import tensorflow as tf


class PoleSolver(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.hidden_layer = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(1, activation=tf.tanh)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        hidden = self.input_layer(inputs)
        hidden = self.hidden_layer(hidden)
        hidden = tf.reshape(hidden, shape=[1, -1])
        output = self.output_layer(hidden)
        return tf.squeeze(output, axis=-1)

def loss_fn(observation, diff_action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5  # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    x, x_dot, theta, theta_dot = observation

    force = force_mag * diff_action

    costheta = tf.cos(theta)
    sintheta = tf.sin(theta)

    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x_dot = x_dot + tau * xacc
    x = x + tau * x_dot
    theta_dot = theta_dot + tau * thetaacc
    theta = theta + tau * theta_dot
    return tf.square(theta)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    solver = PoleSolver()
    optimizer = tf.keras.optimizers.Adam()

    for i_episode in range(100):
        observation = env.reset()
        observation = tf.constant(observation, dtype=tf.float32)
        for t in range(200):
            env.render()
            # print("Input:", observation)
            with tf.GradientTape() as tape:
                tape.watch(observation)
                diff_answer = solver(observation)[0]
                loss = loss_fn(observation, diff_answer)
                print("Loss:", loss)
                grad = tape.gradient(loss, solver.trainable_variables)
                # print(grad)
                optimizer.apply_gradients(zip(grad, solver.trainable_variables))

            answer = diff_answer.numpy()
            answer = 1 if answer > 0 else 0
            observation, reward, done, info = env.step(answer)
            observation = tf.constant(observation, dtype=tf.float32)
            # observation, reward, done, info = env.step(env.action_space.sample())
            # print(reward)
            # print("Output", observation)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
