import itertools
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model

from config import config
from data.dataset import Dataset
from model_runner import ModelRunner
from registry.registry import ModelRegistry, DatasetRegistry
from utils.measure import Timer


def split_batch(predictions, variable_count):  # TODO: This is task specific, move somewhere
    batched_logits = []
    i = 0
    for length in variable_count:
        batched_logits.append(predictions[i:i + length])
        i += length
    return batched_logits


def main():
    model = ModelRegistry().resolve(config.model)()
    dataset = DatasetRegistry().resolve(config.task)()

    optimizer = tfa.optimizers.RectifiedAdam(config.learning_rate,
                                             total_steps=config.train_steps,
                                             warmup_proportion=config.warmup)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    train(dataset, model, optimizer)
    model.summary()
    test(dataset, model, optimizer)


def train(dataset, model: Model, optimizer):
    writer = tf.summary.create_file_writer(config.train_dir)
    ckpt, manager, runner = prepare_model(dataset, model, optimizer)

    mean_loss = tf.metrics.Mean()
    timer = Timer(start=True)
    validation_data = dataset.validation_data()

    # TODO: Check against step in checkpoint
    for features, labels in itertools.islice(dataset.train_data(), config.train_steps):
        loss, gradients = runner.train_step(features, labels)

        mean_loss.update_state(loss)

        if int(ckpt.step) % 100 == 0:
            loss_mean = mean_loss.result()
            with writer.as_default():
                tf.summary.scalar("loss", loss_mean, step=int(ckpt.step))

            print(f"{int(ckpt.step)}. step;\tloss: {loss_mean:.5f};\ttime: {timer.lap_time():.3f}s")
            mean_loss.reset_states()

            with tf.name_scope("gradients"):
                with writer.as_default():
                    for grd, var in zip(gradients, model.trainable_variables):
                        tf.summary.histogram(var.name, grd, step=int(ckpt.step))

            with tf.name_scope("variables"):
                with writer.as_default():
                    for var in model.trainable_variables:  # type: tf.Variable
                        tf.summary.histogram(var.name, var, step=int(ckpt.step))

        if int(ckpt.step) % 1000 == 0:
            mean_acc, mean_total_acc = validate_model(validation_data, runner, dataset.accuracy_fn)
            with tf.name_scope("accuracy"):
                with writer.as_default():
                    tf.summary.scalar("accuracy", mean_acc, step=int(ckpt.step))
                    tf.summary.scalar("total_accuracy", mean_total_acc, step=int(ckpt.step))
            print(f"Validation accuracy: {mean_acc.numpy():.4f}; total accuracy {mean_total_acc.numpy():.4f}")

        if int(ckpt.step) % 1000 == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")

        if int(ckpt.step) % 100 == 0:
            writer.flush()

        ckpt.step.assign_add(1)


def validate_model(data, runner, accuracy_fn):
    mean_acc = tf.metrics.Mean()
    mean_total_acc = tf.metrics.Mean()
    # TODO: Don't use slice here, init dataset only once

    for (features, labels), (variable_count, normal_clauses) in itertools.islice(data, 100):
        prediction = runner.prediction(features, labels)
        prediction = np.round(tf.sigmoid(prediction))
        prediction = split_batch(prediction, variable_count)

        for batch, (pred, clause) in enumerate(zip(prediction, normal_clauses)):
            accuracy = accuracy_fn(pred, clause)

            if accuracy == 1:
                mean_total_acc.update_state(accuracy)
            else:
                mean_total_acc.variables[1].assign_add(1)

            mean_acc.update_state(accuracy)

    return mean_acc.result(), mean_total_acc.result()


def prepare_model(dataset, model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, config.train_dir, max_to_keep=config.ckpt_count)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Initializing new model!")

    runner = ModelRunner(model, dataset.loss_fn, optimizer)

    return ckpt, manager, runner


def test(dataset, model, optimizer):
    ckpt, manager, runner = prepare_model(dataset, model, optimizer)

    mean_acc = tf.metrics.Mean()
    mean_total_acc = tf.metrics.Mean()
    for (features, labels), (variable_count, normal_clauses) in dataset.test_data():  # TODO: This is task specific too
        prediction = runner.prediction(features, labels)
        prediction = split_batch(prediction, variable_count)  # TODO: Move somewhere, this is task specific

        for batch, (pred, clause) in enumerate(zip(prediction, normal_clauses)):
            soft_prediction = tf.sigmoid(pred)
            hard_prediction = np.round(soft_prediction)
            accuracy = dataset.accuracy_fn(hard_prediction, clause)
            if accuracy == 1:
                mean_total_acc.update_state(accuracy)
            else:
                mean_total_acc.variables[1].assign_add(1)

            mean_acc.update_state(accuracy)

    print("Accuracy:", mean_acc.result().numpy())
    print("Total fully correct:", mean_total_acc.result().numpy())


if __name__ == '__main__':
    global config
    config = config.parse_args()

    tf.config.run_functions_eagerly(config.eager)

    if config.restore:
        print(f"Restoring model from last checkpoint in '{config.restore}'!")
        config.train_dir = config.restore
    else:
        current_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
        config.train_dir = config.train_dir + "/" + config.task + "_" + current_date

    main()
