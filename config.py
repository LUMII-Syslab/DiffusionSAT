import argparse

import sys
import subprocess
import json

# We have removed depdendency on registry.registry, since
# it depends on files (e.g., SAT datasets), which may use Config.
# A direct dependency on registry.registry then would enforce circular imports in Python.
# Intead, we call another python3 interpreter to obtain info about registry.registry.
def registry_choices() -> dict:
    python3 = sys.executable # e.g. 'python3'
    result = subprocess.run([python3, 'registry/registry.py'], capture_output=True, text=True)

    if result.returncode == 0:
        # Parse the JSON output from the subprocess
        output_dict = json.loads(result.stdout)
        return output_dict
    else:
        raise Exception("Error parsing dict from registry/registry: "+result.stderr)

class Config:
    """Data and placement config: """
    from pathlib import Path

    train_dir = './np-solver'
    hyperopt_dir = '/host-dir/optuna'
    data_dir = '/host-dir/data'
    force_data_gen = False

    ckpt_count = 3
    eager = False

    restore = None
    label = ""

    #scale_test_batch = True # whether to use the same test data to fill the whole test batch - useful for finding multiple solutions at once (setting by SK)
    # this functionality moved directly to data/diffusion_sat.py

    """Training and task selection config: """
    optimizer = 'radam'
    train_steps = 167000
#    train_steps = 500000
    warmup = 0.0
    learning_rate = 0.0003
    model = 'query_sat'  # query_sat,  query_sat_lit, neuro_sat, tsp_matrix_se
    #task = "splot" # task === dataset; for "splot" use input_mode="variables"
    #task = "satlib"
    #task = "k_sat"
    task = "diffusion-sat"
     # '3-sat'  # k-sat, k_color, 3-sat, clique, primes, sha-gen2019, dominating_set, euclidean_tsp, asymmetric_tsp
     
    # Applicable to SAT-based tasks:
    input_mode = 'literals'  # "variables" or "literals", applicable to SAT
    max_nodes_per_batch = 20_000 # 20_000 for Nvidia T4, 60_000 for more advanced cards (setting by SK)
    #sat_solver_for_generators = "unigen"
    # !!! see: data/diffusion_sat.py#get_sat_solution
      # "default" (means: Glucose4+lingeling), "glucose", "lingeling", "unigen", "quicksampler"
      # see SatSolverRegistry in registry/registry.py for details

    """Supported training and evaluation modes: """
    train = True
    evaluate = True
    test_invariance = False
    evaluate_round_gen = False
    evaluate_batch_gen = False
    evaluate_batch_gen_train = False
    evaluate_variable_gen = False
    test_classic_solver = False
    test_cactus = False

    """Internal config variables: """
    __arguments_parsed = False

    @classmethod
    def parse_config(cls):
        if cls.__arguments_parsed:
            raise RuntimeError("Arguments already parsed!")

        config = cls.__argument_parser().parse_args()
        for key, value in config.__dict__.items():
            setattr(cls, key, value)
        cls.__arguments_parsed = True

    @classmethod
    def __argument_parser(cls):
        choices = registry_choices()
        
        config_parser = argparse.ArgumentParser()

        config_parser.add_argument('--train_dir', type=str, default=cls.train_dir)
        config_parser.add_argument('--data_dir', type=str, default=cls.data_dir)
        config_parser.add_argument('--restore', type=str, default=None)
        #config_parser.add_argument('--restore', type=str, default=Config.train_dir + '/sha-gen2019_22_08_30_08:08:52')
        #config_parser.add_argument('--label', type=str, default=cls.label)

        config_parser.add_argument('--ckpt_count', type=int, default=cls.ckpt_count)
        config_parser.add_argument('--eager', action='store_true', default=cls.eager)

        config_parser.add_argument('--optimizer', default=cls.optimizer, const=cls.optimizer, nargs='?',
                                   choices=["radam"])

        config_parser.add_argument('--train_steps', type=int, default=cls.train_steps)
        config_parser.add_argument('--warmup', type=float, default=cls.warmup)
        config_parser.add_argument('--learning_rate', type=float, default=cls.learning_rate)

        config_parser.add_argument('--model', type=str, default=cls.model, const=cls.model, nargs='?',
                                   choices=choices["ModelRegistry"])

        config_parser.add_argument('--task', type=str, default=cls.task, const=cls.task, nargs='?',
                                   choices=choices["DatasetRegistry"])

        config_parser.add_argument('--sat_solver_for_generators', type=str, default=cls.task, const=cls.task, nargs='?',
                                   choices=choices["SatSolverRegistry"])

        config_parser.add_argument('--input_mode', type=str, default=cls.input_mode, const=cls.input_mode, nargs='?',
                                   choices=['variables', 'literals'])
        
        config_parser.add_argument('--force_data_gen', action='store_true', default=cls.force_data_gen)

        config_parser.add_argument('--train', action='store_true', default=cls.train)
        config_parser.add_argument('--evaluate', action='store_true', default=cls.evaluate)
        config_parser.add_argument('--test_invariance', action='store_true', default=cls.test_invariance)
        config_parser.add_argument('--evaluate_round_gen', action='store_true', default=cls.evaluate_round_gen)
        config_parser.add_argument('--evaluate_batch_gen', action='store_true', default=cls.evaluate_batch_gen)
        config_parser.add_argument('--evaluate_batch_gen_train', action='store_true', default=cls.evaluate_batch_gen_train)
        config_parser.add_argument('--evaluate_variable_gen', action='store_true', default=cls.evaluate_variable_gen)

        return config_parser


if __name__ == "__main__":  # test
    ccc = registry_choices()
    print("choices are:")
    print(ccc)
    