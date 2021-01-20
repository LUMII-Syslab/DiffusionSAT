import os
import time

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.datasets import cifar10

print(tf.__version__)


def create_dataset(data):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255, y))
    # dataset = dataset.repeat()
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(32)
    return dataset


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_dataset = create_dataset((x_train, y_train))
test_dataset = create_dataset((x_test, y_test))


class Solver(Model):
    def __init__(self):
        super(Solver, self).__init__()
        self.conv1 = Conv2D(32, 3, strides=2, activation='relu')
        self.conv2 = Conv2D(32, 3, strides=2, activation='relu')
        self.conv2 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(10)

    @tf.function
    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class Grader(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_layer = tf.keras.layers.Dense(64, activation=tf.nn.relu)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden_layer = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.hidden_layer2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        hidden = self.input_layer(inputs)
        hidden = self.flatten(hidden)
        hidden = self.hidden_layer(hidden)
        hidden = self.hidden_layer2(hidden)
        output = self.output_layer(hidden)
        return tf.squeeze(output, axis=-1)


# Create an instance of the model


grader = Grader()
grader_optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(solver, solver_optimizer, images, labels):
    with tf.GradientTape() as solver_tape, tf.GradientTape() as grader_tape:
        predictions = solver(images)

        one_hot_labels = tf.one_hot(tf.squeeze(labels, axis=-1), 10)

        fake_loss1 = tf.reduce_mean(grader(tf.concat([one_hot_labels, predictions], axis=1), training=True))
        real_loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=predictions))

        random_predictions = tf.random.normal(tf.shape(predictions), stddev=0.25)
        fake_loss2 = tf.reduce_mean(grader(tf.concat([one_hot_labels, random_predictions], axis=1), training=True))
        real_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=random_predictions))

        grader_loss = tf.square(tf.stop_gradient(real_loss1) - fake_loss1) + tf.square(tf.stop_gradient(real_loss2) - fake_loss2)

        solver_loss = fake_loss1

        grader_gradients = grader_tape.gradient(grader_loss, grader.trainable_variables)
        if grader_loss < 0.0001:
            solver_gradients = solver_tape.gradient(solver_loss, solver.trainable_variables)
            solver_optimizer.apply_gradients(zip(solver_gradients, solver.trainable_variables))

        grader_optimizer.apply_gradients(zip(grader_gradients, grader.trainable_variables))

    return solver_loss, grader_loss


test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def test_step(images, labels):
    predictions = solver(images)

    one_hot_labels = tf.one_hot(tf.squeeze(labels, axis=-1), 10)
    fake_loss = grader(tf.concat([one_hot_labels, predictions], axis=1))
    t_loss = tf.reduce_mean(fake_loss)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 100
SUMMARY_DIR = './summary'

summary_writer = tf.summary.create_file_writer(SUMMARY_DIR)

solver = Solver()
solver_optimizer = tf.keras.optimizers.Adam()

reset_model = 1
for epoch in range(EPOCHS):
    # solver = Solver()
    # solver_optimizer = tf.keras.optimizers.Adam()

    start = time.time()
    for steps, (images, labels) in enumerate(train_dataset):
        fake_loss, grader_loss = train_step(solver, solver_optimizer, images, labels)
        # print(f"{steps} Fake loss {fake_loss}, Grader loss {grader_loss}")

    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)

    elapsed = time.time() - start
    print('elapsed: %f' % elapsed)

    template = f'Epoch {epoch + 1}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}'
    print(template)

    # Reset the metrics for the next epoch
    test_loss.reset_states()
    test_accuracy.reset_states()

print('Training Finished.')
