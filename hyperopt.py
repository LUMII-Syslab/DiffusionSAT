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
import numpy as np


def prepare_checkpoints(train_dir, model, optimizer):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, train_dir, max_to_keep=Config.ckpt_count)
    ckpt.restore(manager.latest_checkpoint).expect_partial()

    return ckpt, manager


def running_mean(x, N):
    x = np.pad(x, (N // 2, N - 1 - N // 2), mode='edge')
    return np.convolve(x, np.ones((N,)) / N, mode='valid')


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
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    beta_1 = trial.suggest_float("beta_1", 0.5, 1.0)

    optimizer = AdaBeliefOptimizer(learning_rate, beta_1=beta_1, clip_gradients=True)

    # batch_size = trial.suggest_categorical("batch_size", [5000, 10000, 15000, 20000])
    batch_size = 10000

    model = ModelRegistry().resolve(Config.model)(optimizer=optimizer, trial=trial)
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     max_nodes_per_batch=batch_size,
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

    accuracies = []

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
            metrics = evaluate_metrics(dataset, validation_data, model, steps=150)
            total_accuracy = metrics[0].get_values(reset_state=False)[1].numpy()

            for metric in metrics:
                metric.log_in_tensorboard(reset_state=False, step=int(ckpt.step))
                metric.log_in_stdout(step=int(ckpt.step))

            accuracies.append(total_accuracy)
            trial_accuracy = running_mean(accuracies, 10)[-1]
            trial.report(trial_accuracy, int(ckpt.step))

            # Handle pruning based on the intermediate value.
            # if trial.should_prune():
            #     raise optuna.TrialPruned()

        if int(ckpt.step) % 1000 == 0:
            save_path = ckpt_manager.save()
            print(f"Saved checkpoint for step {int(ckpt.step)}: {save_path}")

        if int(ckpt.step) % 100 == 0:
            writer.flush()

        ckpt.step.assign_add(1)

    return running_mean(accuracies, 15)[-1]


def create_if_doesnt_exist(folder: str):
    hyp_dir = Path(folder)
    if not hyp_dir.exists():
        hyp_dir.mkdir(parents=True)


if __name__ == '__main__':
    config = Config.parse_config()
    tf.config.run_functions_eagerly(Config.eager)

    create_if_doesnt_exist(Config.hyperopt_dir)

    study_name = "query_sat_on_3sat_no_prune3"
    storage = f"sqlite:///{Config.hyperopt_dir}/np_solvers.db"
    runs_folder = Config.hyperopt_dir + "/" + study_name
    Config.train_dir = runs_folder

    create_if_doesnt_exist(runs_folder)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        # pruner=optuna.pruners.HyperbandPruner(min_resource=5000, max_resource=Config.train_steps),
        direction="maximize")

    study.set_user_attr("model", Config.model)
    study.set_user_attr("dataset", Config.task)
    study.set_user_attr("train_steps", Config.train_steps)

    study.optimize(objective_fn, n_trials=50)

    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"{runs_folder}/history.png")

    fig = optuna.visualization.plot_slice(study)
    fig.write_image(f"{runs_folder}/slice.png")

    fig = optuna.visualization.plot_edf(study)
    fig.write_image(f"{runs_folder}/edf.png")

    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"{runs_folder}/importance.png")

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("\t Value: ", trial.value)

    print("\t Params: ")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")
