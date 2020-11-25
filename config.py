import argparse

from registry.registry import ModelRegistry, DatasetRegistry


class Config:
    train_dir = '/host-dir/np-solver'
    data_dir = '/host-dir/data'
    force_data_gen = False

    ckpt_count = 3
    eager = False

    optimizer = 'radam'
    train_steps = 100000
    warmup = 0.0
    learning_rate = 0.0001
    model = 'multistep_tsp'
    task = 'euclidean_tsp'

    restore = None
    label = ""

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

        config_parser.add_argument('--force_data_gen', action='store_true', default=cls.force_data_gen)

        return config_parser
