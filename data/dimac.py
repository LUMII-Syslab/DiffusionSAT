import random
import shutil
from abc import abstractmethod, ABCMeta
from pathlib import Path

import tensorflow as tf
from pysat.solvers import Glucose4

from data.dataset import Dataset
from utils.iterable import elements_to_str, elements_to_int, flatten

from config import Config

def compute_adj_indices(clauses):
    adj_indices_pos = [[v - 1, idx] for idx, c in enumerate(clauses) for v in c if v > 0]
    adj_indices_neg = [[abs(v) - 1, idx] for idx, c in enumerate(clauses) for v in c if v < 0]

    return adj_indices_pos, adj_indices_neg


class SatInstances(metaclass=ABCMeta):
    """ Base dataset for generating SAT instances.
    """
    
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

class TaskSpecifics(metaclass=ABCMeta):
    
    @abstractmethod
    def prepare_dataset(self, batched_dataset: tf.data.Dataset):
        """ Prepare task specifics for dataset.
        :param dataset: tf.data.Dataset with attributes 
                        clauses, solutions, batched_clauses, adj_indices_pos, adj_indices_neg, variable_count, clauses_in_formula, cells_in_formula
        :return: tf.data.Dataset
        """
        pass
    
    @abstractmethod
    def args_for_train_step(self, step_data) -> dict:
        """ Converts the given batch (from previously prepared datased via prepare_dataset) 
            to a dictionary of arguments to be passed to the neural network train_step method.
        """
        pass
    
    
    @abstractmethod
    def metrics(self, initial=False) -> list:
        pass

class BatchedDimacsDataset(Dataset):
    """ Base class for datasets that are based on DIMACS files.
    """

    def __init__(self, sat_instances: SatInstances, sat_specifics: TaskSpecifics, data_dir=Config.data_dir, data_dir_suffix=None, force_data_gen=Config.force_data_gen, max_nodes_per_batch=Config.max_nodes_per_batch,
                 shuffle_size=200, **kwargs) -> None:
        self.sat_instances = sat_instances
        self.sat_specifics = sat_specifics
        self.force_data_gen = force_data_gen
        self.data_dir = Path(data_dir) / self.__class__.__name__
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle_size = shuffle_size
        self.data_dir_suffix = self.__class__.__name__ if data_dir_suffix is None else data_dir_suffix

    # implemented abstract methods:
    def train_data(self) -> tf.data.Dataset:
        data = self.fetch_dataset(self.sat_instances.train_generator, mode="train")
        data = data.shuffle(self.shuffle_size)
        data = data.repeat()
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def validation_data(self) -> tf.data.Dataset:
        if hasattr(self.sat_instances, "validation_generator"):
            data = self.fetch_dataset(self.sat_instances.validation_generator, mode="validation")
        else:
            data = self.fetch_dataset(self.sat_instances.test_generator, mode="validation")
        # data = data.shuffle(self.shuffle_size) # ??
        data = data.repeat()
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def test_data(self) -> tf.data.Dataset:
        return self.fetch_dataset(self.sat_instances.test_generator, mode="test")
    
    def args_for_train_step(self, step_data) -> dict:
        return self.sat_specifics.args_for_train_step(step_data)

    def metrics(self, initial=False) -> list:
        return self.sat_specifics.metrics(initial)

    # other methods:
    def fetch_dataset(self, generator: callable, mode: str):
        dimacs_folder = Path(self.data_dir) / Path("dimacs") / Path(f"{mode}_{self.data_dir_suffix}")
        tfrecords_folder = Path(self.data_dir) / Path("tf_records") / Path(f"{mode}_{self.max_nodes_per_batch}_{self.data_dir_suffix}")

        if self.force_data_gen and tfrecords_folder.exists():
            shutil.rmtree(tfrecords_folder)

        if self.force_data_gen and dimacs_folder.exists():
            shutil.rmtree(dimacs_folder) # by SK: remove also dimacs folder to re-generate .dimacs files

        if not dimacs_folder.exists() and not tfrecords_folder.exists():
            self.write_dimacs_to_file(dimacs_folder, generator, mode)

        if not tfrecords_folder.exists():
            self.dimac_to_data(dimacs_folder, tfrecords_folder)

        data = self.read_dataset(tfrecords_folder)
        return self.sat_specifics.prepare_dataset(data)

    def read_dataset(self, data_folder):
        data_files = [str(d) for d in data_folder.glob("*.tfrecord")]
        data = tf.data.TFRecordDataset(data_files, "GZIP", num_parallel_reads=8)
        data = data.map(self.feature_from_file, tf.data.experimental.AUTOTUNE)
        return data

    def write_dimacs_to_file(self, data_folder: Path, data_generator: callable, mode: str):
        if self.force_data_gen and data_folder.exists():
            shutil.rmtree(data_folder)

        if data_folder.exists():
            print("Not recreating data, as folder already exists!")
            return
        else:
            data_folder.mkdir(parents=True)

        print(f"Generating DIMACS data in '{data_folder}' directory!")
        for idx, (n_vars, clauses, *solution) in enumerate(data_generator()):
            solution = solution[0] if solution and solution[0] else get_sat_solution(clauses, mode)
            solution = [int(x > 0) for x in solution]
            solution = elements_to_str(solution)

            clauses = [elements_to_str(c) for c in clauses]
            file = [f"c sol " + " ".join(solution)]
            file += [f"p cnf {n_vars} {len(clauses)}"]
            file += [f"{' '.join(c)} 0" for c in clauses]

            out_filename = data_folder / f"sat_{n_vars}_{len(clauses)}_{idx}.dimacs"
            with open(out_filename, 'w') as f:
                f.write('\n'.join(file))

            if idx % 1000 == 0:
                print(f"{idx} DIMACS files generated...")

    @staticmethod
    def __read_dimacs_details(file):
        with open(file, 'r') as f:
            for line in f:
                values = line.strip().split()
                if values[0] == "p" and values[1] == "cnf":
                    return int(values[2]), int(values[3])

    @staticmethod
    def shift_variable(x, offset):
        return x + offset if x > 0 else x - offset

    def shift_clause(self, clauses, offset):
        return [[self.shift_variable(x, offset) for x in c] for c in clauses]
    
    def sat_node_count(self, n_vars, n_clauses):
        # Warning: This doesn't match node count if we use variables instead of literals!!!
        return 2*n_vars + n_clauses

    def dimac_to_data(self, dimacs_dir: Path, tfrecord_dir: Path):
        files = [d for d in dimacs_dir.glob("*.dimacs")]
        formula_size = [self.__read_dimacs_details(f) for f in files]

        node_count = [self.sat_node_count(n,m) for (n, m) in formula_size]

        # Put formulas with similar size in same batch
        files = sorted(zip(node_count, files))
        batches = self.__batch_files(files)

        options = tf.io.TFRecordOptions(compression_type="GZIP", compression_level=9)
        print(f"Converting DIMACS data from '{dimacs_dir}' into '{tfrecord_dir}'!")

        if not tfrecord_dir.exists():
            tfrecord_dir.mkdir(parents=True)

        dataset_id = 0
        dataset_filename = tfrecord_dir / f"data_{dataset_id}.tfrecord"
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
                dataset_filename = tfrecord_dir / f"data_{dataset_id}.tfrecord"
                tfwriter = tf.io.TFRecordWriter(str(dataset_filename), options)

        # close the last file...
        tfwriter.flush()
        tfwriter.close()
        print(f"Created {len(batches)} data batches in {tfrecord_dir}...\n")

    def prepare_example(self, batch):
        batched_clauses = []
        cells_in_formula = []
        clauses_in_formula = []
        variable_count = []
        offset = 0
        original_clauses = []
        solutions = []

        for file in batch:
            with open(file, 'r') as f:
                lines = f.readlines()

            solution = elements_to_int(lines[0].strip().split()[2:])
            *_, var_count, clauses_count = lines[1].strip().split()
            var_count = int(var_count)
            clauses_count = int(clauses_count)

            clauses = [elements_to_int(line.strip().split()[:-1]) for line in lines[2:]]

            clauses_in_formula.append(clauses_count)
            original_clauses.append(clauses)
            variable_count.append(var_count)
            solutions.append(solution)
            if var_count != len(solution):
                raise Exception("lengths do not match")
            batched_clauses.extend(self.shift_clause(clauses, offset))
            cells_in_formula.append(sum([len(c) for c in clauses]))
            offset += var_count

        adj_indices_pos, adj_indices_neg = compute_adj_indices(batched_clauses)

        example_map = {
            'solutions': self.__int64_feat(solutions),
            'solutions_rows': self.__int64_feat([len(s) for s in solutions]),
            'clauses': self.__int64_feat(original_clauses),
            'clauses_len_first': self.__int64_feat([len(c) for c in original_clauses]),
            'clauses_len_second': self.__int64_feat([len(x) for c in original_clauses for x in c]),
            'batched_clauses': self.__int64_feat(batched_clauses),
            'batched_clauses_rows': self.__int64_feat([len(x) for x in batched_clauses]),
            'adj_indices_pos': self.__int64_feat(adj_indices_pos), # by SK: 2D matrix ar "pozitīvām" šķautnēm: mainīgais x klauzulas#
            'adj_indices_neg': self.__int64_feat(adj_indices_neg), # by SK: 2D matrix ar "negatīvām" šķautnēm: mainīgais x klauzulas#
            'variable_count': self.__int64_feat(variable_count),
            'clauses_in_formula': self.__int64_feat(clauses_in_formula),
            'cells_in_formula': self.__int64_feat(cells_in_formula)
        }

        return tf.train.Example(features=tf.train.Features(feature=example_map))

    @staticmethod
    def __int64_feat(array):
        int_list = tf.train.Int64List(value=flatten(array))
        return tf.train.Feature(int64_list=int_list)

    def __batch_files(self, files):
        files_size = len(files)
        # filter formulas that will not fit in any batch
        files = [(node_count, filename) for node_count, filename in files]

        dif = files_size - len(files)
        if dif > 0:
            print(f"\n\n WARNING: {dif} formulas were not included in dataset as they exceeded max node count! \n\n")

        batches = []
        current_batch = []
        nodes_in_batch = 0

        for nodes_cnt, filename in files:
            if nodes_cnt + nodes_in_batch <= self.max_nodes_per_batch or nodes_in_batch==0:
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
            'solutions': tf.io.RaggedFeature(dtype=tf.int64, value_key='solutions',
                                             partitions=[tf.io.RaggedFeature.RowLengths('solutions_rows')]),
            'adj_indices_pos': tf.io.VarLenFeature(tf.int64),
            'adj_indices_neg': tf.io.VarLenFeature(tf.int64),
            'variable_count': tf.io.VarLenFeature(tf.int64),
            'clauses_in_formula': tf.io.VarLenFeature(tf.int64),
            'cells_in_formula': tf.io.VarLenFeature(tf.int64),
        }

        parsed = tf.io.parse_single_example(data_record, features)

        return {
            "clauses": tf.cast(parsed['clauses'], tf.int32),
            "solutions": tf.cast(parsed['solutions'], tf.int32),
            "batched_clauses": tf.cast(parsed['batched_clauses'], tf.int32),
            "adj_indices_pos": sparse_to_dense(parsed['adj_indices_pos'], dtype=tf.int64, shape=[-1, 2]), # positive edges: var pieder clause
            "adj_indices_neg": sparse_to_dense(parsed['adj_indices_neg'], dtype=tf.int64, shape=[-1, 2]), # negative edges: neg(var) pieder clause
            "variable_count": sparse_to_dense(parsed['variable_count']),
            "clauses_in_formula": sparse_to_dense(parsed['clauses_in_formula']),
            "cells_in_formula": sparse_to_dense(parsed['cells_in_formula']),
        }


def get_sat_solution(clauses: list, mode: str = None):
    if mode != "test":
        raise Exception("This get_sat_solution should not be called if the generator provides the solutions!")
    with Glucose4(bootstrap_with=clauses) as solver:
        is_sat = solver.solve()
        if not is_sat:
            raise ValueError("Can't get solution for UNSAT clauses")
        return solver.get_model()


def sparse_to_dense(sparse_tensor, dtype=tf.int32, shape=None):
    tensor = tf.sparse.to_dense(sparse_tensor)
    tensor = tf.reshape(tensor, shape) if shape else tensor
    return tf.cast(tensor, dtype=dtype)
