#!/usr/bin/env python3

import os
import shutil
import subprocess
import math
import sys
import random
from data.k_sat import KSATVariables



def randomBinaryString(n):
    s = ""
    for _i in range(n):
        s += str(random.randint(0, 1))
    return s

class SHAGen(KSATVariables):
    """ Dataset with random SAT instances based on the SHA1 algorithm. We use cgen inside.
    """

    CGEN_EXECUTABLE = "./data/cgen"
    TMP_FILE_NAME = "data.tmp"

    def __init__(self, data_dir, force_data_gen=False, **kwargs) -> None:
        super(KSATVariables, self).__init__(data_dir, force_data_gen=force_data_gen, **kwargs)
        self.train_size = 10000 # maximum number of samples; if there are less, we will stop earlier
        self.test_size = 5#00

        #### constraints ####
        self.bits_from = 2
        self.bits_to = 8

        self.pad_with_zeroes = True#False
        self.sha_rounds_from = 2
        self.sha_rounds_to = 5

        self.bits_position_from = 0
        self.bits_position_to = 511
           # must be between 0..511; in case of position overflow (bits position + number of generated bits>512), generated bits will be truncated

        #### the desired number of variables ####
        self.min_vars = 30
        self.max_vars = 40
        self.max_attempts = 100 # how many times we need to check the number of variables to be within the given range before we stop the generator

    def train_generator(self) -> tuple:
        return self.__generator(self.train_size)

    def test_generator(self) -> tuple:
        return self.__generator(self.test_size)

    def __generator(self, size) -> tuple:
        samplesSoFar = 0


        while samplesSoFar<size:
            attempts = 0
            while attempts < self.max_attempts:
                n_bits = random.randint(self.bits_from, self.bits_to)

                sha_rounds = random.randint(self.sha_rounds_from, max(self.sha_rounds_from,self.sha_rounds_to))
                if sha_rounds<1:
                    sha_rounds=1
                if sha_rounds>80:
                    sha_rounds=80

                bits_position = random.randint(self.bits_position_from, max(self.bits_position_from,self.bits_position_to))
                if bits_position<0:
                    bits_position=0
                if bits_position>511:
                    bits_position=511

                
                binstr = randomBinaryString(n_bits)
                bits = binstr
                if self.pad_with_zeroes:
                    bits = ('0'*(bits_position-1)) + bits + ('0'*512)
                    bits = bits[:512] # strip bits to the length of 512
                else:
                    # cloning bits to get a string of length 2*512 or a bit more; we we go to the left of the center to implement a shift
                    bits = (bits * (512//n_bits + 1)) 
                    center=len(bits)
                    bits = bits*2
                    bits = bits[center-bits_position:][:512]

                if os.path.exists(SHAGen.TMP_FILE_NAME):
                    os.remove(SHAGen.TMP_FILE_NAME)
                cmd = SHAGen.CGEN_EXECUTABLE+" encode SHA1 -vM 0b"+bits+" except:"+str(bits_position+1)+".."+str(min(bits_position+n_bits,512))+" -r "+str(sha_rounds)+" "+SHAGen.TMP_FILE_NAME 

                # Launching the process and reading its output
                out = subprocess.check_output(cmd, shell=True, text=True)
                #print(cmd,out) # -- debug

                # Searching for the "CNF: <nvars> var" substring;
                # ok will be true iff <nvars> is between MIN_VARS and MAX_VARS;
                # if not ok, we will delete the file.
                ok = False
                j1 = out.find("CNF:")
                j2 = out.find("var", j1+1)
                if j1>=0 and j2>=0:
                    nvars = int(out[j1+4:j2].strip())
                    ok = nvars >= self.min_vars and nvars <= self.max_vars

                if ok:
                    f = open(SHAGen.TMP_FILE_NAME, 'r') 
                    lines = f.readlines()
                    f.close()
                    os.remove(SHAGen.TMP_FILE_NAME)
                    clauses = []
                    for line in lines:
                        line = line.strip()
                        if len(line)==0 or line[0].isalpha():
                            continue
                        clause = []
                        for s in line.split():
                            i = int(s)
                            if i==0:
                                break # end of clause
                            clause.append(i)
                        clauses.append(clause)
                        
                    yield nvars, clauses
                    samplesSoFar+=1
                    break # while attempts
                
                if os.path.exists(SHAGen.TMP_FILE_NAME):
                    os.remove(SHAGen.TMP_FILE_NAME)
                # if break haven't occurred, try the next attempt:
                attempts+=1

            # after while ended, let's check if we reached the attempt limit
            if attempts == self.max_attempts:
                break # stop the iterator, too many attempts; perhaps, we are not able to generate the desired number of variables according to the given constraints



