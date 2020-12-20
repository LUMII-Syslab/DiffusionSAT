#!/usr/bin/env python3

# Script by SK.

import os
import platform
import random
import subprocess

import glob

from data.k_sat import KSAT



class PrimesGen(KSAT):
    """ Dataset with SAT instances based on integer factorization into 2 primes, each below 1000.
    """

    FETCHED_DATA_DIR = "primes"
    if not os.path.exists(FETCHED_DATA_DIR):
        FETCHED_DATA_DIR = "data/"+FETCHED_DATA_DIR

    def __init__(self, data_dir, force_data_gen=False, **kwargs) -> None:
        super(PrimesGen, self).__init__(data_dir, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 10000  # maximum number of samples; if there are less, we will stop earlier
        self.test_size = 1000

        #### constraints ####

        #### the desired number of variables ####
        self.min_vars = 4
        self.max_vars = 100
        self.max_attempts = 100  # how many times we need to check the number of variables to be within the given range before we stop the generator

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:

        fileList = glob.glob(PrimesGen.FETCHED_DATA_DIR+"/*.dimacs")
        print("LIST",PrimesGen.FETCHED_DATA_DIR+"/*.dimacs",fileList)
        fileIndex = 0

        samplesSoFar = 0

        while samplesSoFar < size:
            attempts = 0
            while attempts < self.max_attempts:

                if fileIndex>len(fileList):
                    attempts = self.max_attempts
                    break

                fileName = fileList[fileIndex]
                fileIndex+=1

                ok = True
                
                f = open(fileName, 'r')
                lines = f.readlines()
                f.close()
                clauses = []
                for line in lines:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    if line[0].isalpha():
                        if line[0]=='p':
                            # parse: "p cnf <nvars>""
                            j1 = line.find("p cnf ")
                            j2 = line.find(" ", j1+6)
                            nvars = int(line[j1+6:j2].strip())
                            ok = nvars >= self.min_vars and nvars <= self.max_vars
                            if not ok:
                                break # do not consider other clauses
                        continue # continue with the next line
                    clause = []
                    for s in line.split():
                        i = int(s)
                        if i == 0:
                            break  # end of clause
                        clause.append(i)
                    clauses.append(clause)

                if ok:
                    yield nvars, clauses
                    samplesSoFar += 1
                    break  # while attempts

                # if break haven't occurred, try the next attempt:
                attempts += 1

            # after while ended, let's check if we reached the attempt limit
            if attempts == self.max_attempts:
                break  # stop the iterator, too many attempts; perhaps, we are not able to generate the desired number of variables according to the given constraints
