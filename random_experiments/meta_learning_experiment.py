import random

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class Learner(Model):
    def __init__(self):
        super(Learner, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(32, activation='relu', use_bias=False)
        self.d2 = Dense(10, use_bias=False)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


class OptimusPrime(Model):
    def __init__(self):
        super(OptimusPrime, self).__init__()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(1)

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)


# Create an instance of the model
model = Learner()
optim_model = OptimusPrime()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

initializer = tf.initializers.lecun_normal()
weight_1 = tf.Variable(initializer([784, 128]), name="weight_1")
weight_2 = tf.Variable(initializer([128, 10]), name="weight_1")

flatten = tf.keras.layers.Flatten()


@tf.function
def train_step(images, labels, update_weights=False):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        with tf.GradientTape() as tape2:
            x = flatten(images)
            x = tf.matmul(x, weight_1)
            x = tf.nn.relu(x)
            predictions = tf.matmul(x, weight_2)

            loss = loss_object(labels, predictions)
            gradients = tape2.gradient(loss, [weight_1, weight_2])

        first_weight = tf.stack([gradients[0], weight_1], axis=-1)
        update_1 = tf.squeeze(optim_model(first_weight), axis=-1)
        second_weight = tf.stack([gradients[1], weight_2], axis=-1)
        update_2 = tf.squeeze(optim_model(second_weight), axis=-1)

        x = flatten(images)
        x = tf.matmul(x, weight_1 + update_1)
        x = tf.nn.relu(x)
        predictions = tf.matmul(x, weight_2 + update_2)
        new_loss = loss_object(labels, predictions)

        weight_1.assign_add(update_1 * 0.01)
        weight_2.assign_add(update_2 * 0.01)

        optim_loss = new_loss - tf.stop_gradient(loss)
        gradients = tape.gradient(optim_loss, optim_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, optim_model.trainable_variables))

    train_loss(optim_loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    x = flatten(images)
    x = tf.matmul(x, weight_1)
    x = tf.nn.relu(x)
    predictions = tf.matmul(x, weight_2)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 100


@tf.function
def update_learner(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        with tf.GradientTape() as tape2:
            x = flatten(images)
            x = tf.matmul(x, weight_1)
            x = tf.nn.relu(x)
            predictions = tf.matmul(x, weight_2)

            loss = loss_object(labels, predictions)
            gradients = tape2.gradient(loss, [weight_1, weight_2])

        first_weight = tf.stack([gradients[0], weight_1], axis=-1)
        update_1 = tf.squeeze(optim_model(first_weight), axis=-1)
        second_weight = tf.stack([gradients[1], weight_2], axis=-1)
        update_2 = tf.squeeze(optim_model(second_weight), axis=-1)

        weight_1.assign_add(update_1)
        weight_2.assign_add(update_2)


for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    update_learner(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
