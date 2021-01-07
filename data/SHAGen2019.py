#!/usr/bin/env python3

import os
import platform
import random
import subprocess
from enum import Enum
from data.k_sat import KSAT


def random_binary_string(n):
    return "".join([str(random.randint(0, 1)) for _ in range(n)])

class SHAGen2019(KSAT):
    """ Dataset with random SAT instances based on the SHA1 algorithm. We use cgen with the parameters similar to SAT Competition 2019.
    """

    CGEN_EXECUTABLE = "./cgen"
    if platform.system() == "Linux":
        CGEN_EXECUTABLE = "./data/cgen_linux64"
        if not os.path.exists(CGEN_EXECUTABLE):
            CGEN_EXECUTABLE = "./cgen_linux64"

    if platform.system() == "Darwin":
        CGEN_EXECUTABLE = "./cgen_mac"
        if not os.path.exists(CGEN_EXECUTABLE):
            CGEN_EXECUTABLE = "./data/cgen_mac"

    TMP_FILE_NAME = "data.tmp"

    def __init__(self, data_dir,
                 min_vars=4, max_vars=1000,
                 force_data_gen=False, **kwargs) -> None:
        super(SHAGen2019, self).__init__(data_dir, min_vars=min_vars,
                                     max_vars=max_vars, force_data_gen=force_data_gen, **kwargs)
        # maximum number of samples; if there are less, we will stop earlier
        self.train_size = 10000
        self.test_size = 1000

        #### constraints ####
        # how many free bits; max 512 free bits 
        self.bits_from = 5
        self.bits_to = 30

        # the number of rounds (max==80 by SHA-1 specs)
        self.sha_rounds_from = 17#17
        self.sha_rounds_to = 17#17#30
        #### the desired number of variables ####
        self.min_vars = min_vars
        self.max_vars = 10000#max_vars
        # how many times we need to check the number of variables to be within the given range before we stop the generator
        self.max_attempts = 100

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:
        samplesSoFar = 0

        while samplesSoFar < size:
            attempts = 0
            while attempts < self.max_attempts:
                n_bits = random.randint(self.bits_from, self.bits_to)

                sha_rounds = random.randint(self.sha_rounds_from, max(
                    self.sha_rounds_from, self.sha_rounds_to))
                if sha_rounds < 1:
                    sha_rounds = 1
                if sha_rounds > 80:
                    sha_rounds = 80

                bits_position = 0

                bitsstr = random_binary_string(512)
                hashstr = random_binary_string(160)
                cmd = SHAGen2019.CGEN_EXECUTABLE + " encode SHA1 -vM 0b" + bitsstr + " except:1.." + str(n_bits) + " -vH 0b"+hashstr+" -r " + str(
                        sha_rounds) + " " + SHAGen2019.TMP_FILE_NAME

                # Launching the process and reading its output
                if os.path.exists(SHAGen2019.TMP_FILE_NAME):
                    os.remove(SHAGen2019.TMP_FILE_NAME)

                out = ""
                try:
                    out = subprocess.check_output(
                        cmd, shell=True, universal_newlines=True)
                except:
                    out = "" # an unsatisfiable formula or an execution error
                #print(cmd)
                #print(cmd,"["+out+"]") # -- debug

                # Searching for the "CNF: <nvars> var" substring;
                # ok will be true iff <nvars> is between MIN_VARS and MAX_VARS;
                # if not ok, we will delete the file.
                ok = False
                j1 = out.find("CNF:")
                j2 = out.find("var", j1 + 1)
                if j1 >= 0 and j2 >= 0:
                    nvars = int(out[j1 + 4:j2].strip())
                    ok = nvars >= self.min_vars and nvars <= self.max_vars

                if ok:
                    f = open(SHAGen2019.TMP_FILE_NAME, 'r')
                    lines = f.readlines()
                    f.close()
                    os.remove(SHAGen2019.TMP_FILE_NAME)
                    clauses = []
                    for line in lines:
                        line = line.strip()
                        if len(line) == 0 or line[0].isalpha():
                            continue
                        clause = []
                        for s in line.split():
                            i = int(s)
                            if i == 0:
                                break  # end of clause
                            clause.append(i)
                        clauses.append(clause)

                    yield nvars, clauses
                    samplesSoFar += 1
                    break  # while attempts

                if os.path.exists(SHAGen2019.TMP_FILE_NAME):
                    os.remove(SHAGen2019.TMP_FILE_NAME)
                # if break haven't occurred, try the next attempt:
                attempts += 1

            # after while ended, let's check if we reached the attempt limit
            if attempts == self.max_attempts:
                break  # stop the iterator, too many attempts; perhaps, we are not able to generate the desired number of variables according to the given constraints
