import itertools
import time

import numpy
import tensorflow as tf

from config import Config
from registry.registry import DatasetRegistry
from utils.sat import walksat, build_dimacs_file
import csv


def test_walksat():
    dataset = DatasetRegistry().resolve(Config.task)(data_dir=Config.data_dir,
                                                     force_data_gen=Config.force_data_gen,
                                                     input_mode=Config.input_mode)

    solved = []
    var_count = []
    time_used = []
    for step_data in itertools.islice(dataset.test_data(), 125):
        clauses = [x.numpy() for x in step_data["normal_clauses"]]
        vars_in_graph = step_data["variables_in_graph"].numpy()

        for iclauses, n_vars in zip(clauses, vars_in_graph):
            dimacs = build_dimacs_file(iclauses, n_vars)
            sat, solution, time_elapsed = walksat(dimacs)
            solved.append(int(sat))
            var_count.append(n_vars)
            time_used.append(time_elapsed)

    rows = [[x, y, z] for x, y, z in zip(var_count, solved, time_used)]
    rows = sorted(rows, key=lambda x: x[0])
    rows = [[y, z] for _, y, z in rows]
    rows = numpy.cumsum(rows, axis=0)

    with open("walksat_cactus.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print("Total", len(solved), "Solved:", sum(solved), "Time used:", sum(time_used))


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
