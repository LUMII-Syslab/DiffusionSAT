import itertools
import time

import tensorflow as tf
from tensorflow.keras import Model

from config import Config
from data.dataset import Dataset
from optimization.AdaBelief import AdaBeliefOptimizer
from registry.registry import ModelRegistry, DatasetRegistry
from utils.measure import Timer


def main():
    # optimizer = tfa.optimizers.RectifiedAdam(config.learning_rate,
    #                                          total_steps=config.train_steps,
    #                                          warmup_proportion=config.warmup)
    # optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    optimizer = AdaBeliefOptimizer(Config.learning_rate, beta_1=0.5, clip_gradients=True)
    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    model = ModelRegistry().resolve(Config.model)(optimizer=optimizer)
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir, force_data_gen=Config.force_data_gen)

    ckpt, manager = prepare_checkpoints(model, optimizer)
    train(dataset, model, ckpt, manager)

    mean_acc, mean_total_acc = calculate_accuracy(dataset, dataset.test_data(), model)
    print(f"Accuracy: {mean_acc.result().numpy():.4f}")
    print(f"Total fully correct: {mean_total_acc.result().numpy():.4f}")


def train(dataset: Dataset, model: Model, ckpt, ckpt_manager):
    writer = tf.summary.create_file_writer(Config.train_dir)
    writer.set_as_default()

    mean_loss = tf.metrics.Mean()
    timer = Timer(start_now=True)
    validation_data = dataset.validation_data()
    train_data = dataset.train_data()

    # runner.print_summary([x for x in itertools.islice(train_data, 1)][0])

    # TODO: Check against step in checkpoint
    for step_data in itertools.islice(train_data, Config.train_steps + 1):  # TODO: Here is slowdown, don't use islice
        tf.summary.experimental.set_step(ckpt.step)

        model_data = dataset.filter_model_inputs(step_data)
        model_output = model.train_step(**model_data)
        loss, gradients = model_output["loss"], model_output["gradients"]
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
            mean_acc, mean_total_acc = calculate_accuracy(dataset, validation_data, model, steps=100)
            with tf.name_scope("accuracy"):
                with writer.as_default():
                    tf.summary.scalar("accuracy", mean_acc, step=int(ckpt.step))
                    tf.summary.scalar("total_accuracy", mean_total_acc, step=int(ckpt.step))
            print(f"Validation accuracy: {mean_acc.numpy():.4f}; total accuracy {mean_total_acc.numpy():.4f}")

            if Config.task == 'euclidean_tsp':
                model_greedy, input_greedy, model_beam, input_beam, input_random = calculate_TSP_metrics(dataset, validation_data, model, steps=100)
                with tf.name_scope("TSP_metrics"):
                    with writer.as_default():
                        tf.summary.scalar("model/greedy", model_greedy, step=int(ckpt.step))
                        tf.summary.scalar("input/greedy", input_greedy, step=int(ckpt.step))
                        tf.summary.scalar("model/beam", model_beam, step=int(ckpt.step))
                        tf.summary.scalar("input/beam", input_beam, step=int(ckpt.step))
                        tf.summary.scalar("input/random", input_random, step=int(ckpt.step))
                print(f"model_greedy: {model_greedy.numpy():.2f}%; "
                      f"input_greedy {input_greedy.numpy():.2f}%; "
                      f"model_beam: {model_beam.numpy():.2f}%; "
                      f"input_beam: {input_beam.numpy():.2f}%; "
                      f"input_random: {input_random.numpy():.2f}%; ")

            if Config.model == 'multistep_tsp':
                iterator = itertools.islice(validation_data, 1)
                for step_data in iterator:
                    model_input = dataset.filter_model_inputs(step_data)
                    model.log_visualizations(**model_input)

        if int(ckpt.step) % 1000 == 0:
            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")

        if int(ckpt.step) % 100 == 0:
            writer.flush()

        ckpt.step.assign_add(1)


def prepare_checkpoints(model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, Config.train_dir, max_to_keep=Config.ckpt_count)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Initializing new model!")

    return ckpt, manager


def calculate_accuracy(dataset: Dataset, data: tf.data.Dataset, model: Model, steps: int = None):
    mean_acc = tf.metrics.Mean()
    mean_total_acc = tf.metrics.Mean()

    iterator = itertools.islice(data, steps) if steps else data

    for step_data in iterator:
        model_input = dataset.filter_model_inputs(step_data)
        output = model.predict_step(**model_input)
        accuracy, total_accuracy = dataset.accuracy(output["prediction"], step_data)
        mean_acc.update_state(accuracy)
        mean_total_acc.update_state(total_accuracy)

    return mean_acc.result(), mean_total_acc.result()


def calculate_TSP_metrics(dataset: Dataset, data: tf.data.Dataset, model: Model, steps: int = None):
    mean_model_greedy = tf.metrics.Mean()
    mean_input_greedy = tf.metrics.Mean()
    mean_model_beam   = tf.metrics.Mean()
    mean_input_beam   = tf.metrics.Mean()
    mean_input_random = tf.metrics.Mean()

    iterator = itertools.islice(data, steps) if steps else data

    for step_data in iterator:
        model_input = dataset.filter_model_inputs(step_data)
        output = model.predict_step(**model_input)
        model_greedy, input_greedy, model_beam, input_beam, input_random = dataset.TSP_metrics(output["prediction"], step_data)
        mean_model_greedy.update_state(model_greedy)
        mean_input_greedy.update_state(input_greedy)
        mean_model_beam.update_state(model_beam)
        mean_input_beam.update_state(input_beam)
        mean_input_random.update_state(input_random)

    return mean_model_greedy.result(), mean_input_greedy.result(), mean_model_beam.result(), mean_input_beam.result(), mean_input_random.result()


if __name__ == '__main__':
    config = Config.parse_config()
    tf.config.run_functions_eagerly(Config.eager)

    if Config.restore:
        print(f"Restoring model from last checkpoint in '{Config.restore}'!")
        Config.train_dir = Config.restore
    else:
        current_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
        label = "_" + Config.label if Config.label else ""
        Config.train_dir = Config.train_dir + "/" + Config.task + "_" + current_date + label

    main()
