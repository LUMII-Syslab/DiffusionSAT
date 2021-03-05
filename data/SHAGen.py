#!/usr/bin/env python3

import os
import platform
import random
import subprocess
from enum import Enum
from data.k_sat import KSAT


def random_binary_string(n):
    return "".join([str(random.randint(0, 1)) for _ in range(n)])


class HashSetting(Enum):
    MESSAGE_ONLY = -1   # sets only random message bits (taking into a consideration all constraints on bits);
                        # non-free bits will be either zeroes (if pad_with_zeroes) or the same generated random bits repeated until all 512 message bits are filled
    MESSAGE_OR_HASH = 0 # 50/50 distribution between MESSAGE_ONLY and HASH_ONLY
    HASH_ONLY = 1       # sets only random hash bits (always of length 160), keeping the whole message free
                        # since it is harder to generate a SATISFIABLE CNF, please, increase the number of rounds and the number of allowed variables
    BOTH = 2            # sets both random message bits and hash bits;
                        # since it is harder to generate a SATISFIABLE CNF, please, increase the number of rounds and the number of allowed variables


class SHAGen(KSAT):
    """ Dataset with random SAT instances based on the SHA1 algorithm. We use cgen inside.
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
        super(SHAGen, self).__init__(data_dir, min_vars=min_vars,
                                     max_vars=max_vars, force_data_gen=force_data_gen, **kwargs)
        # maximum number of samples; if there are less, we will stop earlier
        self.train_size = 10000
        self.test_size = 1000

        #### constraints ####
        self.bits_from = 5
        self.bits_to = 8

        self.pad_with_zeroes = True  # False

        self.bits_position_from = 0
        self.bits_position_to = 511
        # must be between 0..511; in case of position overflow (bits position + number of generated bits>512), generated bits will be truncated

        self.hash_setting = HashSetting.MESSAGE_ONLY
            # for the value HASH_ONLY:
            #    self.bits_* and self.pad_with_zeroes are not used;
            #    please, increase the number of rounds (we recommend at least 5) and self.max_vars (we recommend 10000)
            # for the value BOTH:
            #    please, increase the number of rounds (we recommend at least 20) and self.max_vars (we recommend 10000)

        # the number of rounds (max==80 by SHA-1 specs)
        self.sha_rounds_from = 2
        self.sha_rounds_to = 5
        #### the desired number of variables ####
        self.min_vars = min_vars
        self.max_vars = max_vars
        # how many times we need to check the number of variables to be within the given range before we stop the generator
        self.max_attempts = 100

    def train_generator(self) -> tuple:
        return self._generator(self.train_size)

    def test_generator(self) -> tuple:
        return self._generator(self.test_size)

    def _generator(self, size) -> tuple:
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

                # Determining what to generate: message or hash
                what = None
                if self.hash_setting == HashSetting.HASH_ONLY:
                    what = "H"
                elif self.hash_setting == HashSetting.MESSAGE_ONLY:
                    what = "M"
                elif self.hash_setting == HashSetting.BOTH:
                    what = "B"
                else:
                    if random.randint(0, 1) == 0:
                        what = "H"
                    else:
                        what = "M"

                cmd = None

                # Constructing cmd depending on what...
                if what == "H": # no message bits needed; just a hash
                    hashstr = random_binary_string(160)
                    cmd = SHAGen.CGEN_EXECUTABLE + " encode SHA1 -vH 0b"+hashstr+" -r " + str(
                        sha_rounds) + " " + SHAGen.TMP_FILE_NAME
                else:  # "M" or "B"; message bits needed
                    bits_position = random.randint(self.bits_position_from,
                                                   max(self.bits_position_from, self.bits_position_to))
                    if bits_position < 0:
                        bits_position = 0
                    if bits_position > 511:
                        bits_position = 511

                    binstr = random_binary_string(n_bits)
                    bits = binstr
                    if self.pad_with_zeroes:
                        bits = ('0' * (bits_position - 1)) + bits + ('0' * 512)
                        bits = bits[:512]  # strip bits to the length of 512
                    else:
                        # cloning bits to get a string of length 2*512 or a bit more; we we go to the left of the center to implement a shift
                        bits = (bits * (512 // n_bits + 1))
                        center = len(bits)
                        bits = bits * 2
                        bits = bits[center - bits_position:][:512]

                    if what == "B":
                        hashstr = random_binary_string(160)
                        cmd = SHAGen.CGEN_EXECUTABLE + " encode SHA1 -vM 0b" + bits + " except:" + str(
                            bits_position + 1) + ".." + str(min(bits_position + n_bits, 512)) + " -vH 0b"+hashstr+" -r " + str(
                            sha_rounds) + " " + SHAGen.TMP_FILE_NAME
                    else: # "M"
                        cmd = SHAGen.CGEN_EXECUTABLE + " encode SHA1 -vM 0b" + bits + " except:" + str(
                            bits_position + 1) + ".." + str(min(bits_position + n_bits, 512)) + " -r " + str(
                            sha_rounds) + " " + SHAGen.TMP_FILE_NAME

                # Launching the process and reading its output
                if os.path.exists(SHAGen.TMP_FILE_NAME):
                    os.remove(SHAGen.TMP_FILE_NAME)

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
                    f = open(SHAGen.TMP_FILE_NAME, 'r')
                    lines = f.readlines()
                    f.close()
                    os.remove(SHAGen.TMP_FILE_NAME)
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

                if os.path.exists(SHAGen.TMP_FILE_NAME):
                    os.remove(SHAGen.TMP_FILE_NAME)
                # if break haven't occurred, try the next attempt:
                attempts += 1

            # after while ended, let's check if we reached the attempt limit
            if attempts == self.max_attempts:
                break  # stop the iterator, too many attempts; perhaps, we are not able to generate the desired number of variables according to the given constraints
