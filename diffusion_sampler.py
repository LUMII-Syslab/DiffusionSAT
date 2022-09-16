import numpy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import random
from config import Config
import os
from PIL import Image
import itertools

from metrics.sat_metrics import SATAccuracyTF
from model.query_sat import randomized_rounding_tf, distribution_at_time
from optimization.AdaBelief import AdaBeliefOptimizer
from registry.registry import ModelRegistry, DatasetRegistry

#model_path = default=Config.train_dir + '/3-sat_22_09_12_14:19:54'
model_path = default=Config.train_dir + '/3-sat_22_09_13_09:30:24-official'
#model_path = default=Config.train_dir + '/clique_22_09_14_10:06:22-official'

from model.query_sat import t_power
use_baseline_sampling = True
test_rounds=32
diffusion_steps = 32

np.set_printoptions(linewidth=2000, precision=3, suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def prepare_checkpoints(model):
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), model=model)
    manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=1000)

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        print(f"Model restored from {manager.latest_checkpoint}!")
    else:
        print("Checkpoint not found!")

    return ckpt, manager

optimizer = AdaBeliefOptimizer(Config.learning_rate)
model = ModelRegistry().resolve(Config.model)(optimizer=optimizer, test_rounds = test_rounds)
dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                 force_data_gen=Config.force_data_gen,
                                                 input_mode=Config.input_mode)

ckpt, manager = prepare_checkpoints(model)

def predict(model_input, noisy_num, noise_scale):
    output = model.diffusion_step(model_input['adj_matrix'], model_input['clauses_graph'], model_input['variables_graph'],
                                  model_input['solutions'], noise_scale, noisy_num)
    logits = output["prediction"]
    predictions = tf.sigmoid(logits)
    return predictions, output

def reverse_distribution_step(x, predictions, noise_scale, time_increment):
    k = 0.9
    x_new = x*k+predictions*(1-k)
    return x_new

def reverse_distribution_step_theoretic(x, x0, t, t_increment):
    t1 = tf.math.pow(t, t_power) #because such distribution was used in training
    t2 = tf.math.pow(max(0.0,t- t_increment), t_power)
    x_new = distribution_at_time(x0, t1)
    alpha_t = (1 - t1) / (1 - t2)
    #print("alphat", alpha_t, t- t_increment)
    x_unnormed = distribution_at_time(x, 1 - alpha_t) * x_new
    x = x_unnormed / (tf.reduce_sum(x_unnormed, axis=-1, keepdims=True) + 1e-8)
    return x


def diffusion(N, step_data, verbose=True, prepare_image=True):
    image_data = []
    image_data1 = []
    model_input = dataset.filter_model_inputs(step_data)
    solutions = step_data["solutions"]
    n_vars = solutions.flat_values.shape[0]
    n_graphs = tf.shape(step_data['variables_graph_adj'])[0]
    #print(n_vars)
    x = tf.zeros([n_vars, 2])+0.5
    cum_accuracy = np.zeros(n_graphs)

    for t in range(N):
        noise_scale = 1 - t / N
        x_noisy = randomized_rounding_tf(x)
        if use_baseline_sampling: x = x_noisy
        predictions, model_output = predict(model_input, x_noisy, noise_scale)
        metric = SATAccuracyTF()
        #metric.update_state(model_output, step_data)
        #metric.log_in_stdout()
        accuracy, total_accuracy = metric.accuracy(model_output['prediction'], step_data)

        #x = reverse_distribution_step(x, tf.stack([1-predictions, predictions], axis=1), noise_scale, 1/N)
        x = reverse_distribution_step_theoretic(x, tf.stack([1 - predictions, predictions], axis=1), noise_scale, 1 / N)
        cum_accuracy = np.maximum(cum_accuracy, total_accuracy.numpy())
        if verbose: print("accuracy:", accuracy.numpy(), "cum_accuracy:", np.mean(cum_accuracy), "total_acc", tf.reduce_mean(tf.cast(total_accuracy, tf.float32)).numpy())
        # if use_baseline_sampling:
        #     x = dist.reverse_distribution_step_thoeretic(x, predictions, noise_scale, 1 / N)
        # else:
        #     x = dist.reverse_distribution_step(x, predictions, noise_scale, 1/N)
        if verbose: print(noise_scale)
        if verbose: print(tf.transpose(x).numpy())
        if verbose: print(predictions.numpy())

        image_data.append(predictions.numpy())
        image_data1.append(x[:, 1].numpy())

    if prepare_image:
        fig = plt.figure(figsize=(n_vars * 4 / 100, N * 2 / 100), dpi=100)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         nrows_ncols=(1, 2),  # creates 2x2 grid of axes
                         axes_pad=0,  # pad between axes in inch.
                         )
        grid[0].imshow(image_data)
        grid[1].imshow(image_data1)
        grid[0].axes.xaxis.set_visible(False)
        grid[0].axes.yaxis.set_visible(False)
        grid[1].axes.xaxis.set_visible(False)
        grid[1].axes.yaxis.set_visible(False)
        plt.show()
        im = Image.fromarray(np.array(image_data)*255)
        im = im.convert("L")
        im.save("predictions.png")
        im = Image.fromarray(np.clip(np.array(image_data1)*128+127,0,255))
        im = im.convert("L")
        im.save("latent.png")

    total_accuracy = tf.cast(tf.expand_dims(total_accuracy, axis=-1), tf.float32)
    variables_graph = step_data['variables_graph_adj']
    var_correct = tf.sparse.sparse_dense_matmul(variables_graph, total_accuracy, adjoint_a=True)
    predictions = tf.round(predictions)
    #print(tf.reduce_mean(var_correct).numpy())
    #print(var_correct.shape)

    return np.mean(cum_accuracy), predictions,var_correct[:,0]


def test_latest(N, n_batches):
    iterator = itertools.islice(dataset.test_data(), n_batches) if n_batches else dataset.test_data()
    print_progress = True

    counter = 0
    success_rate = 0
    for step_data in iterator:
        if print_progress and counter % 10 == 0:
            print("Testing batch", counter)
        counter += 1
        success, predictions, var_correct = diffusion(N, step_data, verbose=False, prepare_image=False)
        success_rate+=success

    print("test dataset sucess", success_rate/counter)

def test_n_solutions(N, n_batches, trials, test_nr):
    iterator = itertools.islice(dataset.test_data(), n_batches*test_nr) if n_batches else dataset.test_data()

    for _ in range((test_nr-1)*n_batches): #skip some first batches
        next(iterator)

    fraction = 0
    count=0
    for step_data in iterator:
        count+=1
        f = test_n_solutions_batch(N, step_data, trials)
        fraction+=f

    print("total unique fraction", fraction/count)

# def test_n_solutions(N, n_batches, trials):
#     iterator = itertools.islice(dataset.test_data(), n_batches) if n_batches else dataset.test_data()
#     fraction = 0
#     count=0
#     for step_data in iterator:
#         count+=1
#         f = test_n_solutions_batch(N, step_data, trials)
#         fraction+=f
#
#     print("total unique fraction", fraction/count)

def test_n_solutions_batch(N, step_data, trials):
    last_predictions=None
    last_var_correct = None
    success_rate = 0
    graph_pos = step_data['variables_in_graph']
    graph_pos = tf.cumsum(graph_pos).numpy()
    n_graphs = graph_pos.shape[0]
    print(n_graphs)
    graph_pos = numpy.concatenate([[0], graph_pos])
    solution_sets = [set() for _ in range(n_graphs)]
    correct_graph_counts = np.zeros(n_graphs)

    for trial in range(trials):
        success, predictions, var_correct = diffusion(N, step_data, verbose=False, prepare_image=False)
        success_rate+=success
        #print("success_rate", success)

        variables_graph = step_data['variables_graph_adj']
        graph_nodes = tf.sparse.sparse_dense_matmul(variables_graph, tf.expand_dims(tf.ones_like(predictions), axis=-1))
        correct_graphs0 = tf.sparse.sparse_dense_matmul(variables_graph, tf.expand_dims(var_correct, axis=-1))
        correct_graphs0 = tf.cast(tf.equal(correct_graphs0, graph_nodes), tf.float32)
        #print(correct_graphs0.numpy())

        if last_var_correct is not None:
            correct = var_correct * last_var_correct
            equal_vars = (1-tf.abs(last_predictions-predictions))*correct
            not_equal_vars = tf.abs(last_predictions - predictions) * correct
            equal_fraction = tf.reduce_sum(equal_vars)/tf.maximum(tf.reduce_sum(correct),1.)
            not_equal_fraction = tf.reduce_sum(not_equal_vars) / tf.maximum(tf.reduce_sum(correct), 1.)
            print("correct_vars_fraction", tf.reduce_mean(correct).numpy())
            print("equal_fraction", equal_fraction.numpy())
            print("not_equal_fraction", not_equal_fraction.numpy())

            equal_graphs = tf.sparse.sparse_dense_matmul(variables_graph, tf.expand_dims(equal_vars, axis=-1))
            equal_graphs = tf.cast(tf.equal(equal_graphs, graph_nodes), tf.float32)
            correct_graphs = tf.sparse.sparse_dense_matmul(variables_graph, tf.expand_dims(correct, axis=-1))
            correct_graphs = tf.cast(tf.equal(correct_graphs, graph_nodes), tf.float32)
            equal_graphs_fraction = tf.reduce_sum(equal_graphs) / tf.maximum(tf.reduce_sum(correct_graphs), 1.)
            print("equal_graphs_fraction", equal_graphs_fraction.numpy(), "=",tf.reduce_sum(equal_graphs).numpy(), "/", tf.reduce_sum(correct_graphs).numpy())

        predictions = predictions.numpy()
        for i in range(n_graphs):
            if correct_graphs0[i]==1:
                graph_solution = predictions[graph_pos[i]:graph_pos[i+1]]
                graph_solution = tuple(graph_solution)
                solution_sets[i].add(graph_solution)
                correct_graph_counts[i]+=1

        last_predictions = predictions
        last_var_correct = var_correct

    print("test dataset sucess", success_rate/trials)
    n_unique = [len(solution_sets[i]) for i in range(n_graphs)]
    #print(n_unique)
    uniqueFraction = np.mean(n_unique)/np.mean(correct_graph_counts)
    print("unique fraction",uniqueFraction)
    return uniqueFraction

def evaluate_metrics(prediction_tries=1):
    model.prediction_tries=prediction_tries
    metrics = dataset.metrics(False)
    steps = None#100
    print_progress = True
    iterator = itertools.islice(dataset.test_data(), steps) if steps else dataset.test_data()

    counter = 0
    for step_data in iterator:
        if print_progress and counter % 10 == 0:
            print("Testing batch", counter)
        counter += 1
        model_input = dataset.filter_model_inputs(step_data)
        output = model.predict_step(**model_input)
        for metric in metrics:
            metric.update_state(output, step_data)

    for metric in metrics:
        metric.log_in_stdout()

    return metrics

#test_latest(diffusion_steps,n_batches=1)
test_n_solutions(diffusion_steps,n_batches=10, trials=100, test_nr=7)
#evaluate_metrics()