from pathlib import Path

import numpy as np
import tensorflow as tf
from pysat.formula import CNF
from pysat.solvers import Glucose4

from metrics.base import Metric


class UNSATAccuracyTF(Metric):

    def __init__(self) -> None:
        self.mean_unsat_core = tf.metrics.Mean()
        self.means_unsat_found = tf.metrics.Mean()

    def update_state(self, model_output, step_data):
        is_unsat_core, unsat_found = self.__validat_unsat_core(model_output["unsat_core"], step_data)
        self.mean_unsat_core.update_state(is_unsat_core)
        self.means_unsat_found.update_state(unsat_found)

    def log_in_tensorboard(self, step: int = None, reset_state=True):
        mean_unsat_cores, mean_unsat = self.__calc_accuracy(reset_state)

        with tf.name_scope("unsat_core"):
            tf.summary.scalar("unsat_core", mean_unsat_cores, step=step)
            tf.summary.scalar("unsat_core", mean_unsat, step=step)

    def log_in_stdout(self, step: int = None, reset_state=True):
        mean_unsat_cores, mean_unsat = self.__calc_accuracy(reset_state)
        print(f"Found UNSAT cores: {mean_unsat_cores.numpy():.4f}")
        print(f"Found UNSAT formulas: {mean_unsat.numpy():.4f}\n")

    def log_in_file(self, file: str, prepend_str: str = None, step: int = None, reset_state=True):
        mean_unsat_cores, mean_unsat = self.__calc_accuracy(reset_state)
        lines = [prepend_str + '\n'] if prepend_str else []
        lines.append(f"Found UNSAT cores: {mean_unsat_cores.numpy():.4f}\n")
        lines.append(f"Found UNSAT formulas: {mean_unsat.numpy():.4f}\n")

        file_path = Path(file)
        with file_path.open("a") as file:
            file.writelines(lines)

    def reset_state(self):
        self.mean_unsat_core.reset_states()

    def get_values(self, reset_state=True):
        return self.__calc_accuracy(reset_state)

    def __calc_accuracy(self, reset_state):
        mean_unsat_core = self.mean_unsat_core.result()
        mean_unsat_found = self.means_unsat_found.result()

        if reset_state:
            self.reset_state()

        return mean_unsat_core, mean_unsat_found

    def __validat_unsat_core(self, predicted_core, step_data):
        clauses = step_data["normal_clauses"]
        clauses = [x.numpy() for x in clauses]
        clauses_in_graph = [x.shape[0] for x in clauses]

        predicted_core_batched = tf.round(predicted_core)
        predicted_core = []
        i = 0
        for length in clauses_in_graph:
            predicted_core.append(predicted_core_batched[i:i + length].numpy())
            i += length

        filtered_cores = [cl[core == 1] for cl, core in zip(clauses, predicted_core)]
        cores_found = []
        unsat_found = []
        for core in filtered_cores:
            # print(filtered_cores[0].tolist())
            core_cnf = CNF(from_clauses=core.tolist())
            with Glucose4(bootstrap_with=core_cnf) as solver:
                is_sat = solver.solve()

            unsat_found.append(int(not is_sat))

            if is_sat:
                is_core = False
                cores_found.append(int(is_core))
                continue

            is_core = True
            for exclude in range(core.shape[0]):
                sub_core = np.delete(core, exclude, axis=0)
                sub_core = CNF(from_clauses=sub_core.tolist())
                with Glucose4(bootstrap_with=sub_core) as solver:
                    is_sat = solver.solve()
                if not is_sat:
                    is_core = False
                    break

            cores_found.append(int(is_core))

        return np.mean(cores_found), np.mean(unsat_found)
