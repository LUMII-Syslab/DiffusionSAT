import math
import os

import numpy as np
import tensorflow as tf
from pathlib import Path
import shutil
import random
import subprocess
from data.SHAGen2019 import random_binary_string
from data.dataset import Dataset
import platform

from metrics.anf_metrics import ANFAccuracyTF
from metrics.sat_metrics import StepStatistics


class ANF(Dataset):
    def __init__(self, data_dir, force_data_gen, **kwargs) -> None:
        self.train_size = 50000
        self.test_size = 1000
        self.validation_size = 1000

        self.sha_rounds_from = 17
        self.sha_rounds_to = 17
        self.bits_from = 2
        self.bits_to = 20

        self.max_nodes_per_batch = 10000
        self.shuffle_size = 200
        self.ANF_simplify = True

        self.force_data_gen = force_data_gen
        self.data_dir = Path(data_dir) / self.__class__.__name__
        self.CGEN_EXECUTABLE = "./cgen"
        if platform.system() == "Linux":
            self.CGEN_EXECUTABLE = "./data/cgen_linux64"
            if not os.path.exists(self.CGEN_EXECUTABLE):
                self.CGEN_EXECUTABLE = "./cgen_linux64"

        if platform.system() == "Darwin":
            self.CGEN_EXECUTABLE = "./cgen_mac"
            if not os.path.exists(self.CGEN_EXECUTABLE):
                self.CGEN_EXECUTABLE = "./data/cgen_mac"

        self.TMP_FILE_NAME = "data.anf"
        self.TMP_FILE_NAME1 = "data1.anf"

    def train_data(self) -> tf.data.Dataset:
        data = self.fetch_dataset(self.__generator(self.train_size), mode="train")
        data = data.shuffle(self.shuffle_size)
        data = data.repeat()
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def validation_data(self) -> tf.data.Dataset:
        data = self.fetch_dataset(self.__generator(self.validation_size), mode="validation")
        data = data.shuffle(self.shuffle_size)
        data = data.repeat()
        return data.prefetch(tf.data.experimental.AUTOTUNE)

    def test_data(self) -> tf.data.Dataset:
        return self.fetch_dataset(self.__generator(self.test_size), mode="test")

    def fetch_dataset(self, generator: callable, mode: str):
        dimacs_folder = self.data_dir / "dimacs" / f"{mode}_{self.sha_rounds_to}_{self.bits_to}"
        tfrecords_folder = self.data_dir / "tf_records" / f"{mode}_{self.max_nodes_per_batch}_{self.sha_rounds_to}_{self.bits_to}"

        if self.force_data_gen and tfrecords_folder.exists():
            shutil.rmtree(tfrecords_folder)

        if not dimacs_folder.exists():
            self.write_dimacs_to_file(dimacs_folder, generator)

        dataset_name = os.path.join(tfrecords_folder, "dataset")
        if not tfrecords_folder.exists():
            self.dimac_to_data(dimacs_folder, dataset_name)

        data = self.read_dataset(dataset_name)
        return data

    def read_dataset(self, data_folder):
        element_spec = (tf.TensorSpec(shape=[None], dtype=tf.int32),#ands_index1
                        tf.TensorSpec(shape=[None], dtype=tf.int32),#ands_index2
                        tf.TensorSpec(shape=(), dtype=tf.int64), #n_vars
                        tf.TensorSpec(shape=(), dtype=tf.int64),# n_clauses
                        tf.TensorSpec(shape=(), dtype=tf.int64),  # n_ands
                        tf.TensorSpec(shape=[None], dtype=tf.int32),#solution
                        tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32), # (vars, ands)-clauses matrix
                        tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32), # variables-graph
                        tf.SparseTensorSpec(shape=[None, None], dtype=tf.float32)) # clauses-graph

        data = tf.data.experimental.load(data_folder, element_spec)
        return data

    def write_dimacs_to_file(self, data_folder: Path, data_generator):
        if self.force_data_gen and data_folder.exists():
            shutil.rmtree(data_folder)

        if data_folder.exists():
            print("Not recreating data, as folder already exists!")
            return
        else:
            data_folder.mkdir(parents=True)

        print(f"Generating DIMACS data in '{data_folder}' directory!")
        for idx, (fname, n_vars, n_equations) in enumerate(data_generator):
            out_filename = data_folder / f"sat_{n_vars}_{n_equations}_{idx}.anf"
            shutil.move(fname, out_filename)

            if idx % 1000 == 0:
                print(f"{idx} ANF files generated...")

    def dimac_to_data(self, dimacs_dir: Path, tfrecord_dir: Path):
        files = [d for d in dimacs_dir.glob("*.anf")]
        formula_size = [self.__read_anf_details(f) for f in files]

        node_count = [n + m for (n, m) in formula_size]

        # Put formulas with similar size in same batch
        files = sorted(zip(node_count, files))
        batches = self.__batch_files(files)
        #batches = [[f] for f in files]#TODO: batching

        def gen():
            for idx, batch in enumerate(batches):
                if idx%100==0: print("created ",idx,"batches")
                n_vars, clauses, solution, variable_count, clause_count = self.prepare_batch(batch)
                ands_index1 = []
                ands_index2 = []
                #clauses_index = []
                new_clauses = []
                for clause_id, c in enumerate(clauses):
                    new_clause = []
                    for v_pair in c:
                        if len(v_pair) == 1: new_clause.append(v_pair[0])
                        else:
                            new_clause.append(n_vars+1+len(ands_index1))# ands are appended to single vars. account +1 for zero index
                            ands_index1.append(v_pair[0])
                            ands_index2.append(v_pair[1])
                    new_clauses.append(new_clause)
                n_ands = len(ands_index1)
                clauses_indices, clauses_shape = self.clauses2sparse(n_vars+len(ands_index1), new_clauses)
                # graph_count = tf.shape(variable_count)
                # graph_id = tf.range(0, graph_count[0])
                # variables_mask = tf.repeat(graph_id, variable_count)
                # clauses_mask = tf.repeat(graph_id, clause_count)
                #
                # clauses_enum = tf.range(0, var_shape[1], dtype=tf.int32)
                # c_g_indices = tf.stack([clauses_mask, clauses_enum], axis=1)
                # c_g_indices = tf.cast(c_g_indices, tf.int64)
                # clauses_graph_adj = self.create_adjacency_matrix(c_g_indices,
                #                                                  self.create_shape(tf.cast(graph_count[0], tf.int64),
                #                                                                    var_shape[1]))
                #
                # variables_enum = tf.range(0, var_shape[0], dtype=tf.int32)
                # v_g_indices = tf.stack([variables_mask, variables_enum], axis=1)
                # v_g_indices = tf.cast(v_g_indices, tf.int64)
                # variables_graph_adj = self.create_adjacency_matrix(v_g_indices,
                #                                                    self.create_shape(tf.cast(graph_count[0], tf.int64),
                #                                                                      var_shape[0]))

                yield ands_index1, ands_index2, n_vars, len(clauses), n_ands, solution, clauses_indices, clauses_shape, variable_count, clause_count

        dataset = tf.data.Dataset.from_generator(gen, output_types = (tf.int32, tf.int32, tf.int64, tf.int64, tf.int64, tf.int32, tf.int64, tf.int64, tf.int32, tf.int32))
        dataset = dataset.map(lambda ands_index1, ands_index2, n_vars, n_clauses, n_ands, solution, clauses_indices, clauses_shape, variable_count, clause_count:
                              (ands_index1, ands_index2, n_vars, n_clauses, n_ands, solution,
                               self.create_adjacency_matrix(clauses_indices, clauses_shape), self.create_graph_matrix(variable_count, n_vars),self.create_graph_matrix(clause_count, n_clauses))) #workaround since generator cannot return sparse data
        tf.data.experimental.save(dataset, tfrecord_dir)

        print(f"Created {len(batches)} data batches in {tfrecord_dir}...\n")

    def create_graph_matrix(self, variable_count_in_graph, n_vars):
        graph_count = tf.shape(variable_count_in_graph)
        graph_id = tf.range(0, graph_count[0])
        variables_mask = tf.repeat(graph_id, variable_count_in_graph)
        # clauses_mask = tf.repeat(graph_id, clause_count)
        #
        # clauses_enum = tf.range(0, var_shape[1], dtype=tf.int32)
        # c_g_indices = tf.stack([clauses_mask, clauses_enum], axis=1)
        # c_g_indices = tf.cast(c_g_indices, tf.int64)
        # clauses_graph_adj = self.create_adjacency_matrix(c_g_indices,
        #                                                  self.create_shape(tf.cast(graph_count[0], tf.int64),
        #                                                                    var_shape[1]))

        variables_enum = tf.range(0, n_vars, dtype=tf.int32)
        v_g_indices = tf.stack([variables_mask, variables_enum], axis=1)
        v_g_indices = tf.cast(v_g_indices, tf.int64)
        variables_graph_adj = self.create_adjacency_matrix(v_g_indices,
                                                           self.create_shape(tf.cast(graph_count[0], tf.int64), n_vars))
        return variables_graph_adj

    def clauses2sparse(self, var_count, clauses):
        shape = self.create_shape(var_count+1, len(clauses))# var_count does not include zero-th var, so need +1
        indices = [[v, idx] for idx, c in enumerate(clauses) for v in c]
        return indices, shape

    @staticmethod
    def create_shape(variables, clauses):
        dense_shape = tf.stack([variables, clauses])
        return tf.cast(dense_shape, tf.int64)

    @staticmethod
    def create_adjacency_matrix(indices, dense_shape):
        matrix = tf.sparse.SparseTensor(indices,
                                      tf.ones(tf.shape(indices)[0], dtype=tf.float32),
                                      dense_shape=dense_shape)
        matrix = tf.sparse.reorder(matrix)
        return matrix


    @staticmethod
    def __read_anf_details(file):
        values = str(file).strip().split("_") #use values encoded in filename
        n_vars = values[-3]
        n_equations = values[-2]
        return int(n_vars), int(n_equations)

    def prepare_batch(self, batch):
        total_vars = 0
        batched_clauses = []
        solutions = []
        variable_count = []
        clause_count = []
        for file in batch:
            n_vars, clauses, solution = self.prepare_example(file)
            # clauses_in_formula.append(clauses_count)
            # original_clauses.append(clauses)
            # variable_count.append(var_count)
            # solutions.append(solution)
            batched_clauses.extend(self.shift_clause(clauses, total_vars))
            # cells_in_formula.append(sum([len(c) for c in clauses]))
            solutions.extend(solution)
            total_vars += n_vars
            variable_count.append(n_vars)
            clause_count.append(len(clauses))
        return total_vars, batched_clauses, solutions, variable_count, clause_count

    @staticmethod
    def shift_variable(x, offset):
        return x + offset if x != 0 else 0

    def shift_clause(self, clauses, offset):
        return [[[self.shift_variable(x, offset) for x in xa] for xa in c] for c in clauses]

    def __batch_files(self, files):
        files_size = len(files)
        # filter formulas that will not fit in any batch
        files = [(node_count, filename) for node_count, filename in files if node_count <= self.max_nodes_per_batch]

        dif = files_size - len(files)
        if dif > 0:
            print(f"\n\n WARNING: {dif} formulas was not included in dataset as they exceeded max node count! \n\n")

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

    def prepare_example(self, filename):
        n_vars, clauses, var_id_map = self.read_anf(filename)
        is_sat, solution = self.run_external_solver(filename)
        if len(solution) < n_vars: raise Exception("error in solving")
        if not is_sat: raise Exception("should be satisfiable")
        if self.ANF_simplify:
            solution_new = np.zeros(n_vars)-1
            for ind, x in enumerate(solution):
                if ind+1 in var_id_map:
                    solution_new[var_id_map[ind+1]-1] = x
            solution = solution_new
        return n_vars, clauses, solution

    def __generator(self, size) -> tuple:
        samplesSoFar = 0

        while samplesSoFar < size:
            n_bits = random.randint(self.bits_from, self.bits_to)

            sha_rounds = random.randint(self.sha_rounds_from, max(
                self.sha_rounds_from, self.sha_rounds_to))
            if sha_rounds < 1:
                sha_rounds = 1
            if sha_rounds > 80:
                sha_rounds = 80

            bits_position = 0

            bitsstr = random_binary_string(512)
            bitsstr = "0b" + bitsstr

            cmd = self.CGEN_EXECUTABLE + " encode SHA1 -f ANF -vM " + bitsstr + " except:1.." + str(
                n_bits) + " -vH compute -r " + str(
                sha_rounds) + " " + self.TMP_FILE_NAME

            # Launching the process and reading its output
            if os.path.exists(self.TMP_FILE_NAME): os.remove(self.TMP_FILE_NAME)
            if os.path.exists(self.TMP_FILE_NAME1): os.remove(self.TMP_FILE_NAME1)

            try:
                out = subprocess.check_output(
                    cmd, shell=True, universal_newlines=True)
            except:
                out = ""  # an unsatisfiable formula or an execution error
            # print(cmd)
            #print(cmd,"["+out+"]") # -- debug

            # Searching for the "CNF: <nvars> var" substring;
            j1 = out.find("ANF:")
            j2 = out.find("var", j1 + 1)
            j3 = out.find("equations", j2 + 1)

            if j1 >= 0 and j2 >= 0:
                nvars = int(out[j1 + 4:j2].strip())
                nequations = int(out[j2 + 10:j3].strip())
                #print(nvars, nequations)
            else:
                print("error:",cmd, "[" + out + "]")
                continue
                #raise Exception("error generating ANF")

            if self.ANF_simplify:
                nvars, nequations, out_name = self.run_external_simplify(self.TMP_FILE_NAME, self.TMP_FILE_NAME1)

            if nequations == 0: continue
            yield out_name, nvars, nequations
            samplesSoFar += 1

    def var_id(self, var_name, var_id_map):
        if var_name=='1': return 0 # zero id is reserved for constant 1
        if var_name[0] != 'x': raise Exception("invalid var name ", var_name[0])
        var_id = int(var_name[1:].strip("()"))
        if var_id <= 0: raise Exception("invalid var nr")
        if not self.ANF_simplify:
            return var_id
        else:
            # renumber variables to remove unused ones
            if var_id in var_id_map: return var_id_map[var_id]
            new_var_id = len(var_id_map)+1# variable numbers start from 1
            var_id_map[var_id] = new_var_id
            return new_var_id


    def read_anf(self,fileName):
        """
        return list of clauses
        each clause is a list of pairs, variables in each pair are anded together
        variable ids start from 1, id=0 represents constant "1"
        :param fileName:
        :return:
        """
        n_lits = 0
        var_id_map = {} #mapping id_in_file -> new_id
        text_file = open(fileName, "r")
        lines = text_file.readlines()
        clauses = []
        n_vars = 0

        for clause_id in range(len(lines)):
            line = lines[clause_id]
            line = line.strip()
            if self.ANF_simplify and line.startswith("c Fixed values"):break #skip fixed values and equivalences, they should have been simplified
            if len(line)>0 and line[0] != 'c':
                items = [x.strip() for x in line.split('+')]
                if len(items)>= 1:
                    n = len(items)
                    c =[]
                    for i in range(n):
                        mul_item = items[i]
                        mulsplit = [x.strip() for x in mul_item.split('*')]
                        n_lits += len(mulsplit)
                        mul_list=[self.var_id(v, var_id_map) for v in mulsplit]
                        #if len(mul_list)==1:mul_list.append(0) # add an extra one
                        if len(mul_list)>2 or len(mul_list)==0: raise Exception("limitation: only two ands are allowed")
                        n_vars = max(n_vars, mul_list[0])
                        if len(mul_list)==2: n_vars = max(n_vars, mul_list[1])
                        c.append(mul_list)
                    clauses.append(c)

        #print("max and length=",maxsize,"n_lits=", n_lits, "n_ands=",n_ands, "unique_ands=", len(ands_set))

        text_file.close()
        return n_vars, clauses, var_id_map

    def filter_model_inputs(self, step_data) -> dict:
        return {"ands_index1":step_data[0],
                "ands_index2":step_data[1],
                "n_vars":step_data[2],
                "n_clauses":step_data[3],
                "n_ands": step_data[4],
                "solution":step_data[5],
                "clauses_adj":step_data[6],
                "variables_graph":step_data[7],
                "clauses_graph":step_data[8]}

    def run_external_solver(self, filename, solver_exe: str = "binary/bosphorus-linux64"):
        """
        :param solver_exe: Absolute or relative path to solver executable
        :return: returns True if formula is satisfiable and False otherwise, and solutions in form [1,2,-3, ...]
        """
        exe_path = Path(solver_exe).resolve()
        #"--anfread --xl 0 --sat 0 --el 0 --anfwrite t1"
        input_str = "--anfread "+str(filename)+ " --xl 0 --sat 0 --el 0 --solve" #todo: batching
        output = subprocess.run([str(exe_path)]+input_str.split(" "), stdout=subprocess.PIPE, universal_newlines=True)

        satisfiable = [line for line in output.stdout.split("\n") if line.startswith("s ")]
        if len(satisfiable) > 1:
            raise ValueError("More than one satisifiability line returned!")

        is_sat = satisfiable[0].split()[-1]
        if is_sat != "ANF-SATISFIABLE" and is_sat != "UNSATISFIABLE":
            raise ValueError("Unexpected satisfiability value!")

        is_sat = is_sat == "ANF-SATISFIABLE"

        if is_sat:
            variables = [line[1:].strip() for line in output.stdout.split("\n") if line.startswith("v ")]
            var_list = variables[0].split()
            solution = [1 if "1+" in s else 0 for s in var_list[1:]] #assumes increasing var order
        else:
            solution = []

        return is_sat, solution

    def run_external_simplify(self, filename,out_filename, solver_exe: str = "binary/bosphorus-linux64"):
        """
        :param solver_exe: Absolute or relative path to solver executable
        :return: returns True if formula is satisfiable and False otherwise, and solutions in form [1,2,-3, ...]
        """
        exe_path = Path(solver_exe).resolve()
        #"--anfread --xl 0 --sat 0 --el 0 --anfwrite t1"
        input_str = "--anfread "+filename + " --xl 0 --sat 0 --el 0 --anfwrite " + out_filename
        output = subprocess.run([str(exe_path)]+input_str.split(" "), stdout=subprocess.PIPE, universal_newlines=True)

        satisfiable = [line for line in output.stdout.split("\n") if line.startswith("c Simplifying finished.")]
        if len(satisfiable) == 0:
            print(output)
            raise Exception("error running simplification")

        # c Num equations: 0
        num_equations = [line for line in output.stdout.split("\n") if line.startswith("c Num equations:")]
        if len(num_equations) == 0:
            print(num_equations)
            raise ValueError("invalid simplification output")
        n_equations = int(num_equations[-1].split()[-1])

        num_vars = [line for line in output.stdout.split("\n") if line.startswith("c Num free vars:")]
        if len(num_vars) == 0:
            print(num_vars)
            raise ValueError("invalid simplification output")

        n_vars = int(num_vars[-1].split()[-1])-1

        return n_vars, n_equations, out_filename

    def metrics(self, initial=False) -> list:
        return [ANFAccuracyTF(),StepStatistics()]




