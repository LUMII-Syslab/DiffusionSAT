import itertools
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from utils.AdaBelief import AdaBeliefOptimizer

from config import config
from model_runner import ModelRunner
from registry.registry import ModelRegistry, DatasetRegistry
from utils.measure import Timer


def main():
    model = ModelRegistry().resolve(config.model)()
    dataset = DatasetRegistry().resolve(config.task)()

    # optimizer = tfa.optimizers.RectifiedAdam(config.learning_rate,
    #                                          total_steps=config.train_steps,
    #                                          warmup_proportion=config.warmup)
    #optimizer = tf.keras.optimizers.Adam(config.learning_rate)
    optimizer = AdaBeliefOptimizer(config.learning_rate, clip_gradients=True)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    train(dataset, model, optimizer)
    test(dataset, model, optimizer)


def train(dataset, model: Model, optimizer):
    writer = tf.summary.create_file_writer(config.train_dir)
    writer.set_as_default()
    ckpt, manager, runner = prepare_model(dataset, model, optimizer)

    mean_loss = tf.metrics.Mean()
    timer = Timer(start_now=True)
    validation_data = dataset.validation_data()
    train_data = dataset.train_data()

    #runner.print_summary([x for x in itertools.islice(train_data, 1)][0])

    # TODO: Check against step in checkpoint
    for step_data in itertools.islice(train_data, config.train_steps + 1):
        tf.summary.experimental.set_step(int(ckpt.step))
        loss, gradients = runner.train_step(step_data)

        mean_loss.update_state(loss)

        if int(ckpt.step) % 100 == 0:
            loss_mean = mean_loss.result()
            with writer.as_default():
                tf.summary.scalar("loss", loss_mean, step=int(ckpt.step))

            print(f"{int(ckpt.step)}. step;\tloss: {loss_mean:.5f};\ttime: {timer.lap():.3f}s")
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
            mean_acc, mean_total_acc = validate_model(validation_data, runner, dataset.accuracy)
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


def prepare_model(dataset, model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, config.train_dir, max_to_keep=config.ckpt_count)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Initializing new model!")

    runner = ModelRunner(model, dataset, optimizer)

    return ckpt, manager, runner


def validate_model(data, runner, accuracy_fn):  # TODO: Validation and test is basically same function. Merge them.
    mean_acc = tf.metrics.Mean()
    mean_total_acc = tf.metrics.Mean()

    for step_data in itertools.islice(data, 100):  # TODO: Add this to config
        prediction = runner.prediction(step_data)
        accuracy, total_accuracy = accuracy_fn(prediction, step_data)
        mean_acc.update_state(accuracy)
        mean_total_acc.update_state(total_accuracy)

    return mean_acc.result(), mean_total_acc.result()


def test(dataset, model, optimizer):
    ckpt, manager, runner = prepare_model(dataset, model, optimizer)

    mean_acc = tf.metrics.Mean()
    mean_total_acc = tf.metrics.Mean()
    for step_data in dataset.test_data():
        prediction = runner.prediction(step_data)
        accuracy, total_accuracy = dataset.accuracy(prediction, step_data)
        mean_acc.update_state(accuracy)
        mean_total_acc.update_state(total_accuracy)

    print(f"Accuracy: {mean_acc.result().numpy():.4f}")
    print(f"Total fully correct: {mean_total_acc.result().numpy():.4f}")


if __name__ == '__main__':
    global config
    config = config.parse_args()

    tf.config.run_functions_eagerly(config.eager)

    if config.restore:
        print(f"Restoring model from last checkpoint in '{config.restore}'!")
        config.train_dir = config.restore
    else:
        current_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
        config.train_dir = config.train_dir + "/" + config.task + "_" + current_date + "_" + config.label

    main()
