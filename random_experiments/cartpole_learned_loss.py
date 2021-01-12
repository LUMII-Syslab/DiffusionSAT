import gym
import tensorflow as tf
import numpy as np


class PoleSolver(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = tf.keras.layers.Dense(32, activation=tf.nn.relu)
        self.hidden_layer = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.output_observations = tf.keras.layers.Dense(1, name="observation")
        self.output_action = tf.keras.layers.Dense(1, activation=tf.sigmoid, name="action")

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        hidden = self.input_layer(inputs)
        hidden = self.hidden_layer(hidden)
        output_action = self.output_action(tf.reshape(hidden, [1, -1]))

        return tf.squeeze(output_action, axis=-1)


class PoleGrader(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = tf.keras.layers.Dense(32, activation=tf.nn.leaky_relu)
        self.hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.hidden_layer2 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        inputs = tf.expand_dims(inputs, axis=-1)
        hidden = self.input_layer(inputs)
        hidden = self.hidden_layer(hidden)
        hidden = self.hidden_layer2(hidden)
        hidden = tf.reshape(hidden, shape=[1, -1])
        output = self.output_layer(hidden)
        return tf.squeeze(output, axis=-1)


def grader_loss(grade_real, grade_fake):
    real_loss = cross_entropy(tf.ones_like(grade_real), grade_real)
    fake_loss = cross_entropy(tf.zeros_like(grade_fake), grade_fake)
    total_loss = real_loss + fake_loss
    return total_loss


def solver_loss(grade_fake):
    return cross_entropy(tf.ones_like(grade_fake), grade_fake)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    solver = PoleSolver()
    grader = PoleGrader()
    optimizer_solver = tf.keras.optimizers.Adam(0.0001)
    optimizer_grader = tf.keras.optimizers.Adam(0.0001)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    for i_episode in range(1000):
        observation = env.reset()
        observation = tf.constant(observation, dtype=tf.float32)
        mean_loss = tf.keras.metrics.Mean()
        for t in range(200):
            env.render()
            with tf.GradientTape() as solver_tape, tf.GradientTape() as grader_tape:
                action = solver(observation)

                ### Non-diff start ###
                answer = action.numpy()
                answer = int(np.round(answer)[0])
                # action = env.action_space.sample()
                new_observation, reward, done, info = env.step(answer)
                new_observation = tf.constant(new_observation, dtype=tf.float32)
                ### Non-diff end ###

                pole_location = grader(tf.concat([observation, action], axis=0))

                s_loss = tf.square(pole_location)
                g_loss = tf.square(pole_location - new_observation[2])
                if g_loss > 0.00001:
                    g_grad = grader_tape.gradient(g_loss, grader.trainable_variables)
                    optimizer_grader.apply_gradients(zip(g_grad, grader.trainable_variables))
                else:
                    s_grad = solver_tape.gradient(s_loss, solver.trainable_variables)
                    g_grad = grader_tape.gradient(g_loss, grader.trainable_variables)

                    optimizer_solver.apply_gradients(zip(s_grad, solver.trainable_variables))
                    optimizer_grader.apply_gradients(zip(g_grad, grader.trainable_variables))

                mean_loss.update_state(g_loss)
                print("Solver loss:", s_loss.numpy(), "Grader loss:", g_loss.numpy())
                observation = new_observation
            # observation, reward, done, info = env.step(env.action_space.sample())
            # print(reward)
            # print("Output", observation)
            if done:
                print("Loss:", mean_loss.result().numpy())
                print("Episode finished after {} timesteps".format(t + 1))
                break
    env.close()
