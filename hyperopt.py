import itertools
import time
import uuid
from pathlib import Path

import optuna
import tensorflow as tf
from tensorflow.keras import Model

from config import Config
from data.dataset import Dataset
from metrics.base import EmptyMetric
from optimization.AdaBelief import AdaBeliefOptimizer
from registry.registry import ModelRegistry, DatasetRegistry
from utils.measure import Timer


def prepare_checkpoints(train_dir, model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, train_dir, max_to_keep=Config.ckpt_count)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    return ckpt, manager


def evaluate_metrics(dataset: Dataset, data: tf.data.Dataset, model: Model, steps: int = None, initial=False) -> list:
    metrics = dataset.metrics(initial)
    iterator = itertools.islice(data, steps) if steps else data

    empty = True
    for step_data in iterator:
        model_input = dataset.filter_model_inputs(step_data)
        output = model.predict_step(**model_input)
        for metric in metrics:
            metric.update_state(output, step_data)
        empty = False

    return metrics if not empty else [EmptyMetric()]


def objective_fn(trial):
    current_date = time.strftime("%y_%m_%d_%T", time.gmtime(time.time()))
    train_dir = Config.train_dir + "/" + Config.task + "_" + current_date

    dataset, model, ckpt, manager = prepare_model(trial, train_dir)
    final_accuracy = train(train_dir, trial, dataset, model, ckpt, manager)

    return final_accuracy


def prepare_model(trial: optuna.Trial, train_dir):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    optimizer = AdaBeliefOptimizer(learning_rate, beta_1=0.5, clip_gradients=True)

    model = ModelRegistry().resolve(Config.model)(optimizer=optimizer)
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     force_data_gen=Config.force_data_gen,
                                                     input_mode=Config.input_mode)

    ckpt, manager = prepare_checkpoints(train_dir, model, optimizer)
    return dataset, model, ckpt, manager


def train(train_dir, trial: optuna.Trial, dataset: Dataset, model: Model, ckpt, ckpt_manager):
    writer = tf.summary.create_file_writer(train_dir)
    writer.set_as_default()

    mean_loss = tf.metrics.Mean()
    timer = Timer(start_now=True)
    validation_data = dataset.validation_data()
    train_data = dataset.train_data()

    last_accuracy = -1

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

            with tf.name_scope("variables"):
                with writer.as_default():
                    for var in model.trainable_variables:  # type: tf.Variable
                        tf.summary.histogram(var.name, var, step=int(ckpt.step))

        if int(ckpt.step) % 1000 == 0:
            metrics = evaluate_metrics(dataset, validation_data, model, steps=100, initial=(int(ckpt.step) == 0))
            total_accuracy = metrics[0].get_values()[1].numpy()

            for metric in metrics:
                metric.log_in_tensorboard(reset_state=False, step=int(ckpt.step))
                metric.log_in_stdout(step=int(ckpt.step))

            trial.report(total_accuracy, int(ckpt.step))
            last_accuracy = total_accuracy

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        if int(ckpt.step) % 1000 == 0:
            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")

        if int(ckpt.step) % 100 == 0:
            writer.flush()

        ckpt.step.assign_add(1)

    return last_accuracy


def create_if_doesnt_exist(folder: str):
    hyp_dir = Path(folder)
    if not hyp_dir.exists():
        hyp_dir.mkdir(parents=True)


if __name__ == '__main__':
    config = Config.parse_config()
    tf.config.run_functions_eagerly(Config.eager)

    create_if_doesnt_exist(Config.hyperopt_dir)

    study_name = str(uuid.uuid1())
    storage = f"sqlite:///{Config.hyperopt_dir}/studies.db"
    runs_folder = Config.hyperopt_dir + "/" + study_name
    Config.train_dir = runs_folder

    create_if_doesnt_exist(runs_folder)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.HyperbandPruner(),
        direction="minimize")

    study.optimize(objective_fn, n_trials=100)
