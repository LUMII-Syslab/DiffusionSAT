#!/usr/bin/env python3

import os
import shutil
import subprocess
import math
import sys
import platform
import random

# SETTINGS #

TARGET_DIR = "output"
CLEANUP_TARGET_DIR = True # useful for repetitive runs

# Generate all binary strings of the length:
BITS_FROM = 4
BITS_TO = 6

# Generate the specified number of samples (random bit strings) for EACH length:
# Specify 0 to generate all possible binary strings.
# Please, notice, that some of these samples may be filtered out, according to the number of generated variables (see below).
NUMBER_OF_SAMPLES = 2

NUMBER_OF_SHA_ROUNDS = 2 # between 1 and 80

# When generating bits, whether to pad them with zeroes or copy them to fill all 512 bits:
PAD_WITH_ZEROES = False

# Only output files with the number of generated CNF variables in the following bounds:
MIN_VARS = 20
MAX_VARS = 100


CGEN_EXECUTABLE = "./cgen"
if platform.system()=="Linux":
    CGEN_EXECUTABLE = "./data/cgen_linux64"
    if not os.path.exists(CGEN_EXECUTABLE):
        CGEN_EXECUTABLE = "./cgen_linux64"

if platform.system()=="Darwin":
    CGEN_EXECUTABLE = "./cgen_mac"
    if not os.path.exists(CGEN_EXECUTABLE):
        CGEN_EXECUTABLE = "./data/cgen_mac"

############

def cleanup(dirpath):
    if not os.path.exists(TARGET_DIR):
        return
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def randomBinaryString(n):
    s = ""
    for _i in range(n):
        s += str(random.randint(0, 1))
    return s

if CLEANUP_TARGET_DIR:
    print("Cleaning up "+TARGET_DIR+"...")
    cleanup(TARGET_DIR)

hexDigits = math.ceil(BITS_TO / 4)
decDigits = math.ceil( math.log(BITS_TO,10.0) )

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

for N_BITS in range(BITS_FROM, BITS_TO+1):
    print("Processing "+str(N_BITS)+" bits...")    
    rng = None    
    if (NUMBER_OF_SAMPLES==0):
        rng = range(0,2**N_BITS) # all possible bit values
    else:
        rng = range(0, NUMBER_OF_SAMPLES)

    
    for i in rng:
        curBits = format(i, '0'+str(N_BITS)+'b')
        if NUMBER_OF_SAMPLES==0:
            curBits = randomBinaryString(N_BITS)

        revBits = curBits[::-1]
        bits = None
        if PAD_WITH_ZEROES:
            bits = revBits + (  ('0'*N_BITS)*(512//N_BITS) )
        else:
            bits = revBits * (512//N_BITS + 1)
        bits = bits[:512] # strip bits to the length of 512

        fileName = TARGET_DIR+"/sha1_"+str(N_BITS).zfill(decDigits)+"bits_"+format(i,"0"+str(hexDigits)+"X")+".cnf"
        cmd = CGEN_EXECUTABLE+" encode SHA1 -vM 0b"+bits+" except:1.."+str(N_BITS)+" -r "+str(NUMBER_OF_SHA_ROUNDS)+" "+fileName # > /dev/null"
        #-- print(cmd)

        # Launching the process and reading its output
        out = subprocess.check_output(cmd, shell=True, text=True)

        # Searching for the "CNF: <nvars> var" substring;
        # ok will be true iff <nvars> is between MIN_VARS and MAX_VARS;
        # if not ok, we will delete the file.
        ok = False
        j1 = out.find("CNF:")
        j2 = out.find("var", j1+1)
        if j1>=0 and j2>=0:
            nvars = int(out[j1+4:j2].strip())
            ok = nvars >= MIN_VARS and nvars <= MAX_VARS
        
        if not ok:
            os.remove(fileName)
    
print("Done.")
