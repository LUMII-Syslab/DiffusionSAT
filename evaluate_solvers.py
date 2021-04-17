import csv
import itertools
import time

import tensorflow as tf

from config import Config
from registry.registry import DatasetRegistry
from utils.sat import walksat, build_dimacs_file
from utils.visualization import create_cactus_data


def test_walksat():
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     force_data_gen=Config.force_data_gen,
                                                     input_mode=Config.input_mode)

    solved = []
    var_count = []
    time_used = []
    for step, step_data in enumerate(dataset.test_data()):
        clauses = [x.numpy() for x in step_data["normal_clauses"]]
        vars_in_graph = step_data["variables_in_graph"].numpy()

        if step >= 10:
            for iclauses, n_vars in zip(clauses, vars_in_graph):
                dimacs = build_dimacs_file(iclauses, n_vars)
                sat, solution, time_elapsed = walksat(dimacs)
                solved.append(int(sat))
                var_count.append(n_vars)
                time_used.append(time_elapsed)

    rows = create_cactus_data(solved, time_used, var_count)

    with open("walksat_cactus.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


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

    test_walksat()
