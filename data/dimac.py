import random
import shutil
from abc import abstractmethod
from pathlib import Path

import tensorflow as tf

from data.dataset import Dataset
from utils.iterable import elements_to_str, elements_to_int, flatten


def compute_adj_indices(clauses):
    adj_indices_pos = [[v - 1, idx] for idx, c in enumerate(clauses) for v in c if v > 0]
    adj_indices_neg = [[abs(v) - 1, idx] for idx, c in enumerate(clauses) for v in c if v < 0]

    return adj_indices_pos, adj_indices_neg


class DIMACDataset(Dataset):
    """ Base class for datasets that are based on DIMACS files.
    """

    def __init__(self, data_dir, force_data_gen=False, max_nodes_per_batch=5000, shuffle_size=200, **kwargs) -> None:
        self.force_data_gen = force_data_gen
        self.data_dir = Path(data_dir) / self.__class__.__name__
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle_size = shuffle_size

        self.dimacs_dir_name = "dimacs"
        self.data_dir_name = "data"

    @abstractmethod
    def train_generator(self) -> tuple:
        """ Generator function (instead of return use yield), that returns single instance to be writen in DIMACS file.
        This generator should be finite (in the size of dataset)
        :return: tuple(variable_count: int, clauses: list of tuples)
        """
        pass

    @abstractmethod
    def test_generator(self) -> tuple:
        """ Generator function (instead of return use yield), that returns single instance to be writen in DIMACS file.
        This generator should be finite (in the size of dataset)
        :return: tuple(variable_count: int, clauses: list of tuples)
        """
        pass

    @abstractmethod
    def prepare_dataset(self, dataset: tf.data.Dataset):
        """ Prepare task specifics for dataset.
        :param dataset: tf.data.Dataset
        :return: tf.data.Dataset
        """
        pass

    def train_data(self) -> tf.data.Dataset:
        data = self.fetch_dataset(self.train_generator, mode="train")
        data = data.shuffle(self.shuffle_size)
        data = data.repeat()
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def validation_data(self) -> tf.data.Dataset:
        data = self.fetch_dataset(self.test_generator, mode="validation")
        data = data.shuffle(self.shuffle_size)
        data = data.repeat()
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def test_data(self) -> tf.data.Dataset:
        return self.fetch_dataset(self.test_generator, mode="test")

    def fetch_dataset(self, generator: callable, mode: str):
        data_folder = self.data_dir / mode

        if self.force_data_gen and data_folder.exists():
            shutil.rmtree(data_folder)

        if not data_folder.exists():
            self.write_dimacs_to_file(data_folder, generator)
            self.dimac_to_data(data_folder)

        data = self.read_dataset(data_folder)
        return self.prepare_dataset(data)

    def read_dataset(self, data_folder):
        data_folder = data_folder / self.data_dir_name
        data_files = [str(d) for d in data_folder.glob("*.tfrecord")]

        data = tf.data.TFRecordDataset(data_files, "GZIP")
        return data.map(lambda rec: self.feature_from_file(rec), tf.data.experimental.AUTOTUNE)

    def write_dimacs_to_file(self, data_folder: Path, data_generator: callable):
        output_folder = data_folder / self.dimacs_dir_name

        if self.force_data_gen and output_folder.exists():
            shutil.rmtree(output_folder)

        if output_folder.exists():
            print("Not recreating data, as folder already exists!")
            return
        else:
            output_folder.mkdir(parents=True)

        print(f"Generating DIMACS data in '{output_folder}' directory!")
        for idx, (n_vars, clauses) in enumerate(data_generator()):
            clauses = [elements_to_str(c) for c in clauses]
            file = [f"p cnf {n_vars} {len(clauses)}"]
            file += [f"{' '.join(c)} 0" for c in clauses]

            out_filename = output_folder / f"sat_{n_vars}_{len(clauses)}_{idx}.dimacs"
            with open(out_filename, 'w') as f:
                f.write('\n'.join(file))

            if idx % 1000 == 0:
                print(f"{idx} DIMACS files generated...")

    @staticmethod
    def __read_dimacs_details(file):
        with open(file, 'r') as f:
            first_line = f.readline()
            first_line = first_line.strip()
            *_, var_count, clauses_count = first_line.split()
            return int(var_count), int(clauses_count)

    @staticmethod
    def shift_variable(x, offset):
        return x + offset if x > 0 else x - offset

    def shift_clause(self, clauses, offset):
        return [[self.shift_variable(x, offset) for x in c] for c in clauses]

    def dimac_to_data(self, folder: Path):
        dimacs = folder / self.dimacs_dir_name

        files = [d for d in dimacs.glob("*.dimacs")]
        formula_size = [self.__read_dimacs_details(f) for f in files]

        # TODO: Doesn't match our node count as we use variables instead of literals
        node_count = [2 * n + m for (n, m) in formula_size]

        # Put formulas with similar size in same batch
        files = sorted(zip(node_count, files))
        batches = self.__batch_files(files)

        options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=9)

        data_folder = folder / self.data_dir_name
        print(f"Converting DIMACS data from '{dimacs}' into '{data_folder}'!")

        if not data_folder.exists():
            data_folder.mkdir(parents=True)

        dataset_id = 0
        dataset_filename = data_folder / f"data_{dataset_id}.tfrecord"
        batches_in_file = 200
        tfwriter = tf.io.TFRecordWriter(str(dataset_filename), options)

        for idx, batch in enumerate(batches):
            batch_data = self.prepare_example(batch)
            tfwriter.write(batch_data.SerializeToString())
            if idx % batches_in_file == 0:
                print(f"{idx} batches ready...")
                tfwriter.flush()
                tfwriter.close()
                dataset_id += 1
                dataset_filename = data_folder / f"data_{dataset_id}.tfrecord"
                tfwriter = tf.io.TFRecordWriter(str(dataset_filename), options)

        print(f"Created {len(batches)} data batches in {data_folder}...\n")

    def prepare_example(self, batch):
        batched_clauses = []
        cells_in_formula = []
        clauses_in_formula = []
        variable_count = []
        offset = 0
        original_clauses = []

        for file in batch:
            with open(file, 'r') as f:
                lines = f.readlines()

            *_, var_count, clauses_count = lines[0].strip().split()
            var_count = int(var_count)
            clauses_count = int(clauses_count)

            clauses = [elements_to_int(line.strip().split()[:-1]) for line in lines[1:]]

            clauses_in_formula.append(clauses_count)
            original_clauses.append(clauses)
            variable_count.append(var_count)
            batched_clauses.extend(self.shift_clause(clauses, offset))
            cells_in_formula.append(sum([len(c) for c in clauses]))
            offset += var_count

        adj_indices_pos, adj_indices_neg = compute_adj_indices(batched_clauses)

        example_map = {
            'clauses': self.__int64_feat(original_clauses),
            'clauses_len_first': self.__int64_feat([len(c) for c in original_clauses]),
            'clauses_len_second': self.__int64_feat([len(x) for c in original_clauses for x in c]),
            'batched_clauses': self.__int64_feat(batched_clauses),
            'batched_clauses_rows': self.__int64_feat([len(x) for x in batched_clauses]),
            'adj_indices_pos': self.__int64_feat(adj_indices_pos),
            'adj_indices_neg': self.__int64_feat(adj_indices_neg),
            'variable_count': self.__int64_feat(variable_count),
            'clauses_in_formula': self.__int64_feat(clauses_in_formula),
            'cells_in_formula': self.__int64_feat(cells_in_formula)
        }

        return tf.train.Example(features=tf.train.Features(feature=example_map))

    @staticmethod
    def __int64_feat(array):
        int_list = tf.train.Int64List(value=flatten(array))
        return tf.train.Feature(int64_list=int_list)

    def __batch_files(self, files):  # TODO: This is no good as formulas in batches never changes
        # filter formulas that will not fit in any batch
        files = [(node_count, filename) for node_count, filename in files if node_count <= self.max_nodes_per_batch]

        batches = []
        current_batch = []
        nodes_in_batch = 0

        for nodes_cnt, filename in files:
            if nodes_cnt + nodes_in_batch <= self.max_nodes_per_batch:
                current_batch.append(filename)
                nodes_in_batch += nodes_cnt
            else:
                batches.append(current_batch)
                current_batch = []
                nodes_in_batch = 0

        if current_batch:
            batches.append(current_batch)

        random.shuffle(batches)
        return batches

    @tf.function
    def feature_from_file(self, data_record):
        features = {
            'clauses': tf.io.RaggedFeature(dtype=tf.int64, value_key='clauses',
                                           partitions=[tf.io.RaggedFeature.RowLengths("clauses_len_first"),
                                                       tf.io.RaggedFeature.RowLengths("clauses_len_second")]),
            'batched_clauses': tf.io.RaggedFeature(dtype=tf.int64, value_key="batched_clauses",
                                                   partitions=[tf.io.RaggedFeature.RowLengths('batched_clauses_rows')]),
            'adj_indices_pos': tf.io.VarLenFeature(tf.int64),
            'adj_indices_neg': tf.io.VarLenFeature(tf.int64),
            'variable_count': tf.io.VarLenFeature(tf.int64),
            'clauses_in_formula': tf.io.VarLenFeature(tf.int64),
            'cells_in_formula': tf.io.VarLenFeature(tf.int64)
        }

        parsed = tf.io.parse_single_example(data_record, features)

        return {
            "clauses": tf.cast(parsed['clauses'], tf.int32),
            "batched_clauses": tf.cast(parsed['batched_clauses'], tf.int32),
            "adj_indices_pos": sparse_to_dense(parsed['adj_indices_pos'], dtype=tf.int64, shape=[-1, 2]),
            "adj_indices_neg": sparse_to_dense(parsed['adj_indices_neg'], dtype=tf.int64, shape=[-1, 2]),
            "variable_count": sparse_to_dense(parsed['variable_count']),
            "clauses_in_formula": sparse_to_dense(parsed['clauses_in_formula']),
            "cells_in_formula": sparse_to_dense(parsed['cells_in_formula'])
        }


def sparse_to_dense(sparse_tensor, dtype=tf.int32, shape=None):
    tensor = tf.sparse.to_dense(sparse_tensor)
    tensor = tf.reshape(tensor, shape) if shape else tensor
    return tf.cast(tensor, dtype=dtype)
