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
    # optimizer = tfa.optimizers.RectifiedAdam(Config.learning_rate,
    #                                          total_steps=Config.train_steps,
    #                                          warmup_proportion=Config.warmup)
    # optimizer = tf.keras.optimizers.Adam(config.learning_rate)

    optimizer = AdaBeliefOptimizer(Config.learning_rate, beta_1=0.5, clip_gradients=True)
    # optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)  # check for accuracy issues!

    model = ModelRegistry().resolve(Config.model)(optimizer=optimizer)
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     force_data_gen=Config.force_data_gen,
                                                     input_mode=Config.input_mode)

    ckpt, manager = prepare_checkpoints(model, optimizer)

    if Config.train:
        train(dataset, model, ckpt, manager)

    if Config.evaluate:
        test_metrics = evaluate_metrics(dataset, dataset.test_data(), model)
        for metric in test_metrics:
            metric.log_in_stdout()

    if Config.test_invariance:
        test_invariance(dataset, dataset.test_data(), model, 20)


def train(dataset: Dataset, model: Model, ckpt, ckpt_manager):
    writer = tf.summary.create_file_writer(Config.train_dir)
    writer.set_as_default()

    mean_loss = tf.metrics.Mean()
    timer = Timer(start_now=True)
    validation_data = dataset.validation_data()
    train_data = dataset.train_data()

    # TODO: Check against step in checkpoint
    for step_data in itertools.islice(train_data, Config.train_steps + 1):
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

            # with tf.name_scope("gradients"):
            #     with writer.as_default():
            #         for grd, var in zip(gradients, model.trainable_variables):
            #             tf.summary.histogram(var.name, grd, step=int(ckpt.step))

            with tf.name_scope("variables"):
                with writer.as_default():
                    for var in model.trainable_variables:  # type: tf.Variable
                        tf.summary.histogram(var.name, var, step=int(ckpt.step))

        if int(ckpt.step) % 1000 == 0:

            metrics = evaluate_metrics(dataset, validation_data, model, steps=100)
            for metric in metrics:
                metric.log_in_tensorboard(reset_state=False, step=int(ckpt.step))
                metric.log_in_stdout(step=int(ckpt.step))

            if Config.task == 'euclidean_tsp':  # TODO: Make it similar to metrics
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

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Initializing new model!")

    return ckpt, manager


def evaluate_metrics(dataset: Dataset, data: tf.data.Dataset, model: Model, steps: int = None) -> list:
    metrics = dataset.metrics()
    iterator = itertools.islice(data, steps) if steps else data

    for step_data in iterator:
        model_input = dataset.filter_model_inputs(step_data)
        output = model.predict_step(**model_input)
        for metric in metrics:
            metric.update_state(output, step_data)

    return metrics


def test_invariance(dataset: Dataset, data: tf.data.Dataset, model: Model, steps: int = None):
    metrics = dataset.metrics()
    iterator = itertools.islice(data, steps) if steps else data

    for step_data in iterator:
        print("Positive literals: ", tf.math.count_nonzero(tf.clip_by_value(step_data['clauses'], 0, 1).values).numpy())
        print("Negative literals: ", tf.math.count_nonzero(tf.clip_by_value(step_data['clauses'], -1, 0).values).numpy())

        print("\n")
        invariance_original(dataset, metrics, model, step_data)
        print("\n")
        invariance_shuffle_literals(dataset, metrics, model, step_data)
        print("\n")
        invariance_shuffle_clauses(dataset, metrics, model, step_data)
        print("\n")
        invariance_inverse(dataset, metrics, model, step_data)
        print("---------------------------\n")

    return metrics


def invariance_shuffle_literals(dataset, metrics, model, step_data):
    step_data['clauses'] = tf.map_fn(lambda x: tf.random.shuffle(x), step_data["clauses"])
    model_input = dataset.filter_model_inputs(step_data)
    output = model.predict_step(**model_input)
    print("Shuffle literals inside clauses:")
    for metric in metrics:
        metric.update_state(output, step_data)
        metric.log_in_stdout()


def invariance_shuffle_clauses(dataset, metrics, model, step_data):
    clauses = step_data['clauses'].to_tensor()
    clauses = tf.random.shuffle(clauses)
    step_data['clauses'] = tf.RaggedTensor.from_tensor(clauses, padding=0, row_splits_dtype=tf.int32)
    model_input = dataset.filter_model_inputs(step_data)
    output = model.predict_step(**model_input)
    print("Shuffle clauses:")
    for metric in metrics:
        metric.update_state(output, step_data)
        metric.log_in_stdout()


def invariance_inverse(dataset, metrics, model, step_data):
    step_data["adjacency_matrix_pos"], step_data["adjacency_matrix_neg"] = step_data["adjacency_matrix_neg"], step_data["adjacency_matrix_pos"]
    step_data["clauses"] = step_data["clauses"] * -1
    step_data["normal_clauses"] = step_data["normal_clauses"] * -1
    model_input = dataset.filter_model_inputs(step_data)
    output = model.predict_step(**model_input)
    print("Inverse literals:")
    for metric in metrics:
        metric.update_state(output, step_data)
        metric.log_in_stdout()


def invariance_original(dataset, metrics, model, step_data):
    model_input = dataset.filter_model_inputs(step_data)
    output = model.predict_step(**model_input)
    for metric in metrics:
        print("Original values:")
        metric.update_state(output, step_data)
        metric.log_in_stdout()


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
