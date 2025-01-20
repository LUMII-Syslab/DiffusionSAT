import tensorflow as tf
from optimization.AdaBelief import AdaBeliefOptimizer
from config import Config
from model.query_sat import QuerySAT
from utils.DimacsFile import DimacsFile
from data.diffusion_sat_instances import DiffusionSatDataset
from utils.VariableAssignment import VariableAssignment

import numpy as np

from model.query_sat import randomized_rounding_tf, distribution_at_time
from metrics.sat_metrics import SATAccuracyTF
from model.query_sat import t_power

import sys

use_baseline_sampling = True
test_rounds = 32
diffusion_steps = 32# 300 #32
test_unigen = False
self_supervised = False

def reverse_distribution_step(x, predictions, noise_scale, time_increment):
    k = 0.9
    x_new = x * k + predictions * (1 - k)
    return x_new


def reverse_distribution_step_theoretic(x, x0, t, t_increment):
    t1 = tf.math.pow(t, t_power)  # because such distribution was used in training
    t2 = tf.math.pow(max(0.0, t - t_increment), t_power)
    x_new = distribution_at_time(x0, t1)
    alpha_t = (1 - t1) / (1 - t2)
    # print("alphat", alpha_t, t- t_increment)
    x_unnormed = distribution_at_time(x, 1 - alpha_t) * x_new
    x = x_unnormed / (tf.reduce_sum(x_unnormed, axis=-1, keepdims=True) + 1e-8)
    return x


def predict(model, model_input, noisy_num, noise_scale, denoised_num=None):
    if denoised_num is None:
        output = model.diffusion_step(
            model_input["adj_matrix"],
            model_input["clauses_graph"],
            model_input["variables_graph"],
            model_input["solutions"],
            noise_scale,
            noisy_num,
        )
    else:
        output = model.diffusion_step_self(
            model_input["adj_matrix"],
            model_input["clauses_graph"],
            model_input["variables_graph"],
            model_input["solutions"],
            noise_scale,
            noisy_num,
            tf.expand_dims(denoised_num, axis=-1),
        )

    logits = output["prediction"]
    predictions = tf.sigmoid(logits)
    return predictions, output

def __raggedtensor_to_list(tf_arr):
        dims = len(np.shape(tf_arr))
        if dims == 1:
            return tf_arr.numpy().tolist()
        
        n = np.shape(tf_arr)[0]
        result = []
        for i in range(n):
            subarr = tf_arr[i]
            subarr = __raggedtensor_to_list(subarr)                
            result.append(subarr)
        return result

def diffusion(N, model, dataset, step_data, verbose=True, prepare_image=True):
    image_data = []
    image_data1 = []
    model_input = dataset.args_for_train_step(step_data)
    solutions = step_data["solutions"]
    n_vars = solutions.flat_values.shape[0]
    n_graphs = tf.shape(step_data["variables_graph_adj"])[0]
    # print(n_vars)
    x = tf.zeros([n_vars, 2]) + 0.5
    x_use_fixed_step = [-1]*n_vars
    x_fixed = [0]*n_vars
    cum_accuracy = np.zeros(n_graphs)
    predictions = None

    print("KEYS:",list(step_data.keys()),"NVARS:",n_vars)
    print("CLAUSES",step_data["clauses"])
    print("N CLAUSES",step_data["normal_clauses"])
    #print("N CLAUSES SHAPE", tf.shape(step_data["normal_clauses"]))
    print("SOLUTIONS",step_data["solutions"]) # some precomputed solutions
    print("V IN G", step_data["variables_in_graph"])
    print("#CLAUSES ", tf.shape(step_data["clauses"])[0:1])
    # for sol, normal_clauses in zip(step_data["solutions"],step_data["normal_clauses"]):
    #     asgn = VariableAssignment(clauses=normal_clauses)
    #     asgn.assign_all_from_bit_list(sol)
    #     print(sol)
    #     print("int=",int(asgn),asgn.satisfiable())

    for t in range(N):
        noise_scale = 1 - t / N
        x_noisy = randomized_rounding_tf(x)
        if use_baseline_sampling:
            x = x_noisy
        if self_supervised:
            predictions, model_output = predict(
                model, model_input, x_noisy, noise_scale, predictions
            )
        else:
            predictions, model_output = predict(
                model, model_input, x_noisy, noise_scale, denoised_num=None
            )
        # by SK: predictions are with sigmoid; model_output['prediction'] is without
        metric = SATAccuracyTF()
        # metric.update_state(model_output, step_data)
        # metric.log_in_stdout()
        accuracy, total_accuracy = metric.accuracy(
            model_output["prediction"], step_data
        )

        # x = reverse_distribution_step(x, tf.stack([1-predictions, predictions], axis=1), noise_scale, 1/N)
        x = reverse_distribution_step_theoretic(
            x, tf.stack([1 - predictions, predictions], axis=1), noise_scale, 1 / N
        )
        cum_accuracy = np.maximum(cum_accuracy, total_accuracy.numpy())
        if verbose:
            print(
                "accuracy:",
                accuracy.numpy(),
                "cum_accuracy:",
                np.mean(cum_accuracy),
                "total_acc",
                tf.reduce_mean(tf.cast(total_accuracy, tf.float32)).numpy(),
            )
        # if use_baseline_sampling:
        #     x = dist.reverse_distribution_step_thoeretic(x, predictions, noise_scale, 1 / N)
        # else:
        #     x = dist.reverse_distribution_step(x, predictions, noise_scale, 1/N)
        if verbose:
            print(noise_scale)
        if verbose:
            print(tf.transpose(x).numpy())
        if verbose:
            print(predictions.numpy())


        print("PREDICTIONS",predictions.numpy(),tf.shape(predictions))

        xx = tf.round(predictions)

        shift = 0
        for cur_clauses, cur_n in zip(step_data["normal_clauses"], step_data["variables_in_graph"]):
            if x_use_fixed_step[shift]>=0:
                # skip this graph, since we have already found the solution for it
                shift += cur_n
                continue
            asgn = VariableAssignment(n_vars=cur_n, clauses=cur_clauses)
            x_values = xx[shift:shift+cur_n]
            x_bool_values = [bool(b) for b in x_values]
            asgn.assign_all(x_bool_values)
            #print("Samplis: ", str(asgn), asgn.satisfiable())
            if asgn.satisfiable():
                x_fixed[shift:shift+cur_n] = x_values
                x_use_fixed_step[shift:shift+cur_n] = [t]*cur_n
            shift += cur_n

        image_data.append(predictions.numpy())
        image_data1.append(x[:, 1].numpy())


    total_accuracy = tf.cast(tf.expand_dims(total_accuracy, axis=-1), tf.float32)
    variables_graph = step_data["variables_graph_adj"]
    var_correct = tf.sparse.sparse_dense_matmul(
        variables_graph, total_accuracy, adjoint_a=True
    )

    predictions = tf.round(predictions).numpy()
    for i in range(len(x_use_fixed_step)):
        if x_use_fixed_step[i]>=0:
            predictions[i] = x_fixed[i]
    print("use fixed", x_use_fixed_step)

    # print(tf.reduce_mean(var_correct).numpy())
    # print(var_correct.shape)

    return np.mean(cum_accuracy), predictions, var_correct[:, 0]



class DiffusionSampler():

    def __init__(self, model_path, dimacs_filename):
        optimizer = AdaBeliefOptimizer(learning_rate=Config.learning_rate)
        self.model = QuerySAT(
            optimizer=optimizer, test_rounds=test_rounds
        )
        print("model_path is ", model_path)
        self.ckpt, self.manager = self._prepare_checkpoints(self.model, model_path)
        test_dimacs = DimacsFile(filename=dimacs_filename)
        test_dimacs.load()
        self.dataset = DiffusionSatDataset(
            test_dimacs=test_dimacs,
            test_solutions_multiplier_k=10,
            force_data_gen = True
        )
        self.data = self.dataset.test_data()
        self.data = self.data.repeat()
        self.data.prefetch(tf.data.experimental.AUTOTUNE)

    def _prepare_checkpoints(self, model, model_path):
        ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64), model=model
        )  # reading weights
        manager = tf.train.CheckpointManager(ckpt, model_path, max_to_keep=1000)

        ckpt.restore(manager.latest_checkpoint).expect_partial()
        if manager.latest_checkpoint:
            print(f"Model restored from {manager.latest_checkpoint}!")
        else:
            print("Checkpoint not found!")

        return ckpt, manager

    def samples(self, n_samples): 
        """
        :param n_samples: how many correct samples to generate
        :return: returns the dict: solution as int => count
        """

        diffusion_dict = {}
        diffusion_total_samples = 0
        diffusion_sat_samples = 0

        samples_still_needed = n_samples
        success_rate = 0

        counter = 0
        for step_data in self.data:
            if samples_still_needed==0: # repeating step_data until we reach the desired number of SAT samples
                break

            counter += 1
            print("STEP", counter)
            n_graphs = len(step_data["variables_in_graph"])
            n_batched_clauses = tf.shape(step_data["clauses"])[0]
            n_clauses = int(n_batched_clauses / n_graphs)
            n_batched_vars = tf.shape(step_data["variables_graph_adj"])[1]
            n_vars = int(n_batched_vars / n_graphs)
            print("N graphs=",n_graphs)
            print("N clauses=",n_clauses)        
            print(tf.shape(step_data["clauses"])[0])
            print(step_data["clauses"])
            clauses = step_data["clauses"][:n_clauses] # take the clauses from the first graph in the batch


            if diffusion_total_samples>0 and diffusion_sat_samples / diffusion_total_samples < 0.005: # - diffusion_sat_samples > 1000
                print("too many unsat samples; stopping diffusion")
                break  # too many unsat samples

            print(f"diffusion for the remaining {samples_still_needed} samples...")

            success, predictions, var_correct = diffusion(
                diffusion_steps, self.model, self.dataset, step_data, verbose=False, prepare_image=False
            )

            

            print(f"diffusion done.")
            asgn = VariableAssignment(clauses=clauses)

            print("PREDICTIONS")
            print(predictions)
            # if randomize batches:
            # perm = list(range(n_graphs))
            # random.shuffle(perm)
            # for i in perm:

            for i in range(n_graphs):
            #variant:
            #for i in range(1): # take only the first (0-st) graph
            
                #print("diffusion i ",i)
                #print("predictions ", predictions[i*n_vars:(i+1)*n_vars])
                asgn.assign_all_from_bit_list(predictions[i*n_vars:(i+1)*n_vars])

                i_sample = int(asgn)
                is_sat = asgn.satisfiable()

                #print("solution ","{:20d}".format(i_sample),str(asgn)," is_sat=",is_sat)

                diffusion_total_samples += 1
                if is_sat:
                    print("ADDING SAMPLE ",n_samples - samples_still_needed)
                    diffusion_sat_samples += 1
                    if not i_sample in diffusion_dict:
                        diffusion_dict[i_sample] = 0
                    diffusion_dict[i_sample] += 1
                    samples_still_needed -= 1
                    print("  ADDED: ",n_samples - samples_still_needed)
                    print("  TO_ADD: ", samples_still_needed)
                    if samples_still_needed == 0:
                        break


        print("success rate: ", diffusion_sat_samples/diffusion_total_samples)
        return diffusion_dict
            
