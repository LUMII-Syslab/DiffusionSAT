import argparse

from registry.registry import ModelRegistry, DatasetRegistry


class Config:
    """Data and placement config: """
    train_dir = '/host-dir/np-solver'
    hyperopt_dir = '/host-dir/optuna'
    data_dir = '/host-dir/data'
    force_data_gen = False

    ckpt_count = 3
    eager = False

    restore = None
    label = ""

    """Training and task selection config: """
    optimizer = 'radam'
    train_steps = 500000
    warmup = 0.0
    learning_rate = 0.0002
    model = 'neurocore'  # query_sat,  query_sat_lit, neuro_sat, tsp_matrix_se
    task = '3-sat'  # k_sat, k_color, 3-sat, clique, primes, sha-gen2019, dominating_set, euclidean_tsp, asymmetric_tsp
    input_mode = 'literals'  # "variables" or "literals", applicable to SAT

    """Supported training and evaluation modes: """
    train = True
    evaluate = True
    test_invariance = False
    evaluate_round_gen = False
    evaluate_batch_gen = False
    evaluate_batch_gen_train = False
    evaluate_variable_gen = False
    test_classic_solver = False

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
        config_parser = argparse.ArgumentParser()

        config_parser.add_argument('--train_dir', type=str, default=cls.train_dir)
        config_parser.add_argument('--data_dir', type=str, default=cls.data_dir)
        config_parser.add_argument('--restore', type=str, default=None)
        config_parser.add_argument('--label', type=str, default=cls.label)

        config_parser.add_argument('--ckpt_count', type=int, default=cls.ckpt_count)
        config_parser.add_argument('--eager', action='store_true', default=cls.eager)

        config_parser.add_argument('--optimizer', default=cls.optimizer, const=cls.optimizer, nargs='?',
                                   choices=["radam"])

        config_parser.add_argument('--train_steps', type=int, default=cls.train_steps)
        config_parser.add_argument('--warmup', type=float, default=cls.warmup)
        config_parser.add_argument('--learning_rate', type=float, default=cls.learning_rate)

        config_parser.add_argument('--model', type=str, default=cls.model, const=cls.model, nargs='?',
                                   choices=ModelRegistry().registered_names)

        config_parser.add_argument('--task', type=str, default=cls.task, const=cls.task, nargs='?',
                                   choices=DatasetRegistry().registered_names)

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
