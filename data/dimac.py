import shutil
from abc import abstractmethod
from pathlib import Path

import tensorflow as tf

from data.dataset import Dataset


def elements_to_str(inputs: iter):
    return [str(x) for x in inputs]


DATA_FILE_NAME = "feature_data.tfrecord"


class DIMACDataset(Dataset):

    def __init__(self, data_dir, force_data_gen=False, **kwargs) -> None:
        self.data_dir = Path(data_dir) / self.__class__.__name__
        self.force_data_gen = force_data_gen

    @abstractmethod
    def dimacs_generator(self) -> tuple:
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
        data_folder = self.data_dir / 'train'
        self.generate_data(data_folder)
        data = self.read_dataset(data_folder)
        data = self.prepare_dataset(data)
        data = data.shuffle(100)  # TODO: Shuffle size in config
        data = data.repeat()
        data = data.prefetch(100)
        return data

    def validation_data(self) -> tf.data.Dataset:
        data_folder = self.data_dir / 'validation'
        self.generate_data(data_folder)
        data = self.read_dataset(data_folder)
        data = self.prepare_dataset(data)  # type: tf.data.Dataset
        data = data.shuffle(100)  # TODO: Shuffle size in config
        data = data.repeat()
        data = data.prefetch(100)
        return data

    def test_data(self) -> tf.data.Dataset:
        data_folder = self.data_dir / 'test'
        self.generate_data(data_folder)
        data = self.read_dataset(data_folder)
        return self.prepare_dataset(data)

    def read_dataset(self, data_folder):
        tf_record_file = data_folder / DATA_FILE_NAME
        data = tf.data.TFRecordDataset([str(tf_record_file)], "GZIP")  # TODO: Generate several record files
        return data.map(lambda rec: self.feature_from_file(rec), tf.data.experimental.AUTOTUNE)

    def generate_data(self, folder: Path):
        if self.force_data_gen and folder.exists():
            shutil.rmtree(folder)

        if not folder.exists():
            self.write_dimacs_to_file(folder)
            self.dimac_to_data(folder)

    def write_dimacs_to_file(self, folder: Path):
        output_folder = folder / "dimacs"

        if self.force_data_gen and output_folder.exists():
            shutil.rmtree(output_folder)

        if output_folder.exists():
            print("Not recreating data, as folder already exists")
            return
        else:
            output_folder.mkdir(parents=True)

        print(f"Generating data in '{output_folder}' directory!")
        for idx, (n_vars, clauses) in enumerate(self.dimacs_generator()):
            clauses = [elements_to_str(c) for c in clauses]
            file = [f"p cnf {n_vars} {len(clauses)}"]
            file += [f"{' '.join(c)} 0" for c in clauses]

            out_filename = output_folder / f"sat_{n_vars}_{len(clauses)}_{idx}.dimacs"
            with open(out_filename, 'w') as f:
                f.write('\n'.join(file))

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
        dimacs = folder / "dimacs"

        files = [d for d in dimacs.glob("*.dimacs")]
        formula_size = [self.__read_dimacs_details(f) for f in files]

        # TODO: Doesn't match our node count as we use variables instead of literals
        node_count = [2 * n + m for (n, m) in formula_size]

        # Put formulas with similar size in same batch
        files = sorted(zip(node_count, files))
        batches = self.__batch_files(files)

        tf_record_file = folder / DATA_FILE_NAME
        options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=9)

        with tf.io.TFRecordWriter(str(tf_record_file), options) as tfwriter:
            for batch in batches:
                feature = self.prepare_example(batch)
                tfwriter.write(feature.SerializeToString())

    @staticmethod
    def flatten(array: list):
        return [x for c in array for x in c]

    @staticmethod
    def elements_to_int(array: iter):
        return [int(x) for x in array]

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

            clauses = [self.elements_to_int(line.strip().split()[:-1]) for line in lines[1:]]

            clauses_in_formula.append(clauses_count)
            original_clauses.append(clauses)
            variable_count.append(var_count)
            batched_clauses.extend(self.shift_clause(clauses, offset))
            cells_in_formula.append(sum([len(c) for c in clauses]))
            offset += var_count

        adj_indices_pos, adj_indices_neg = self.compute_adj_indices(batched_clauses)

        clauses_len_first = [len(c) for c in original_clauses]
        clauses_len_second = [len(x) for c in original_clauses for x in c]
        clauses = tf.train.Int64List(value=self.flatten(self.flatten(original_clauses)))
        clauses_len_first = tf.train.Int64List(value=clauses_len_first)
        clauses_len_second = tf.train.Int64List(value=clauses_len_second)

        batched_clauses_row_len = [len(x) for x in batched_clauses]
        batched_clauses = tf.train.Int64List(value=self.flatten(batched_clauses))
        batched_clauses_row_len = tf.train.Int64List(value=batched_clauses_row_len)

        adj_indices_pos = tf.train.Int64List(value=self.flatten(adj_indices_pos))
        adj_indices_neg = tf.train.Int64List(value=self.flatten(adj_indices_neg))

        variable_count = tf.train.Int64List(value=variable_count)
        clauses_in_formula = tf.train.Int64List(value=clauses_in_formula)
        cells_in_formula = tf.train.Int64List(value=cells_in_formula)

        example_map = {
            'clauses': tf.train.Feature(int64_list=clauses),
            'clauses_len_first': tf.train.Feature(int64_list=clauses_len_first),
            'clauses_len_second': tf.train.Feature(int64_list=clauses_len_second),
            'batched_clauses': tf.train.Feature(int64_list=batched_clauses),
            'batched_clauses_rows': tf.train.Feature(int64_list=batched_clauses_row_len),
            'adj_indices_pos': tf.train.Feature(int64_list=adj_indices_pos),
            'adj_indices_neg': tf.train.Feature(int64_list=adj_indices_neg),
            'variable_count': tf.train.Feature(int64_list=variable_count),
            'clauses_in_formula': tf.train.Feature(int64_list=clauses_in_formula),
            'cells_in_formula': tf.train.Feature(int64_list=cells_in_formula)
        }

        features = tf.train.Features(feature=example_map)
        return tf.train.Example(features=features)

    @staticmethod
    def compute_adj_indices(clauses):
        adj_indices_pos = []
        adj_indices_neg = []

        for clause_id, clause in enumerate(clauses):
            for var in clause:
                if var > 0:
                    adj_indices_pos.append([var - 1, clause_id])
                elif var < 0:
                    adj_indices_neg.append([abs(var) - 1, clause_id])
                else:
                    raise ValueError("Variable can't be 0 in the DIMAC format!")

        return adj_indices_pos, adj_indices_neg

    @staticmethod
    def __batch_files(files):  # TODO: This is no good as formulas in batches never changes
        max_nodes_per_batch = 5000  # TODO: Put this in better place

        # filter formulas that will not fit in any batch
        files = [(node_count, filename) for node_count, filename in files if node_count <= max_nodes_per_batch]

        batches = []
        current_batch = []
        nodes_in_batch = 0

        for nodes_cnt, filename in files:
            if nodes_cnt + nodes_in_batch <= max_nodes_per_batch:
                current_batch.append(filename)
                nodes_in_batch += nodes_cnt
            else:
                batches.append(current_batch)
                current_batch = []
                nodes_in_batch = 0

        if current_batch:
            batches.append(current_batch)

        return batches

    @staticmethod
    def feature_from_file(data_record):
        parsed = tf.io.parse_single_example(data_record, {
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
        })

        adj_indices_pos = tf.sparse.to_dense(parsed['adj_indices_pos'])
        adj_indices_pos = tf.reshape(adj_indices_pos, shape=[-1, 2])
        adj_indices_neg = tf.sparse.to_dense(parsed['adj_indices_neg'])
        adj_indices_neg = tf.reshape(adj_indices_neg, shape=[-1, 2])

        variable_count = tf.sparse.to_dense(parsed['variable_count'])
        clauses_in_formula = tf.sparse.to_dense(parsed['clauses_in_formula'])
        cells_in_formula = tf.sparse.to_dense(parsed['cells_in_formula'])

        output = {
            "clauses": parsed['clauses'],
            "batched_clauses": parsed['batched_clauses'],
            "adj_indices_pos": adj_indices_pos,
            "adj_indices_neg": adj_indices_neg,
            "variable_count": variable_count,
            "clauses_in_formula": clauses_in_formula,
            "cells_in_formula": cells_in_formula
        }

        return output
