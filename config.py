import argparse

from registry.registry import ModelRegistry, DatasetRegistry

config = argparse.ArgumentParser()

config.add_argument('--train_dir', type=str, default='/host-dir/np-solver')
config.add_argument('--data_dir', type=str, default='/host-dir/data')
config.add_argument('--restore', type=str, default=None)

config.add_argument('--ckpt_count', type=int, default=3)
config.add_argument('--eager', action='store_true', default=False)

config.add_argument('--optimizer', default='radam', const='radam', nargs='?', choices=["radam"])
config.add_argument('--train_steps', type=int, default=100000)
config.add_argument('--warmup', type=float, default=0.0)
config.add_argument('--learning_rate', type=float, default=0.00001)

config.add_argument('--model', type=str, default='neuro_sat', const='query_sat', nargs='?',
                    choices=ModelRegistry().registered_names)
config.add_argument('--task', type=str, default='random_sat', const='random_sat', nargs='?',
                    choices=DatasetRegistry().registered_names)
