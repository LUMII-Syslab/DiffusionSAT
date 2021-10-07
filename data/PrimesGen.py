#!/usr/bin/env python3

# Script by SK.

import glob
import os

from data.k_sat import KSAT
from utils.sat import remove_unused_vars, remove_useless_clauses
import random

class PrimesGen(KSAT):
    """ Dataset with SAT instances based on integer factorization into 2 primes, each below 1000.
    """

    FETCHED_DATA_DIR = "primes"
    if not os.path.exists(FETCHED_DATA_DIR):
        FETCHED_DATA_DIR = "data/" + FETCHED_DATA_DIR

    def __init__(self, data_dir, min_vars=4, max_vars=200, force_data_gen=False, **kwargs) -> None:
        super(PrimesGen, self).__init__(data_dir, min_vars=min_vars, max_vars=max_vars, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 20000  # maximum number of samples; if there are less, we will stop earlier
        self.test_size = 1000

        #### constraints ####

        #### the desired number of variables ####
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.max_attempts = 100000000  # TODO: remove attempts
        self.file_list = glob.glob(PrimesGen.FETCHED_DATA_DIR + "/*.dimacs")
        random.shuffle(self.file_list)
        # print("LIST", PrimesGen.FETCHED_DATA_DIR + "/*.dimacs", fileList)

    def train_generator(self) -> tuple:
        return self._generator1(self.train_size, self.file_list[self.test_size:])

    def test_generator(self) -> tuple:
        return self._generator1(self.test_size, self.file_list[:self.test_size])

    def _generator1(self, size, file_list) -> tuple:

        file_index = 0
        samples_so_far = 0

        while samples_so_far < size:
            attempts = 0
            while attempts < self.max_attempts:

                if file_index >= len(file_list):
                    attempts = self.max_attempts
                    break

                file_name = file_list[file_index]
                file_index += 1

                ok = True

                f = open(file_name, 'r')
                lines = f.readlines()
                f.close()
                clauses = []
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line[0].isalpha():
                        if line[0] == 'p':
                            # parse: "p cnf <nvars>""
                            *_, nvars, clauses_n = line.strip().split(" ")
                            nvars = int(nvars)
                            ok = self.min_vars <= nvars <= self.max_vars
                            if not ok:
                                break  # do not consider other clauses
                        continue  # continue with the next line
                    clause = []
                    for s in line.split():
                        i = int(s)
                        if i == 0:
                            break  # end of clause
                        clause.append(i)
                    clauses.append(clause)

                if ok:
                    clauses = remove_useless_clauses(clauses)
                    yield remove_unused_vars(nvars, clauses)
                    samples_so_far += 1
                    break  # while attempts

                # if break haven't occurred, try the next attempt:
                attempts += 1

            # after while ended, let's check if we reached the attempt limit
            if attempts == self.max_attempts:
                break  # stop the iterator, too many attempts; perhaps, we are not able to generate the desired number of variables according to the given constraints
