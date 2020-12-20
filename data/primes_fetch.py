#!/usr/bin/env python3

# Script by SK.
# Please, do not launch this script! The data have been already fetched. Ask SK.

import os
import shutil
import subprocess
import math
import sys
import platform
import random

# SETTINGS #

TARGET_DIR = "primes"

MAX_PRIME = 1000

############

print("Please, do not launch this script! The data have been already fetched. Ask SK.")
sys.exit(0)

if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

PRIMES = [2]

for i in range(3,MAX_PRIME+1,2):
    isPotentialPrime = True
    j = 0
    while j<len(PRIMES) and PRIMES[j]*PRIMES[j]<=i:
        if i%PRIMES[j]==0:
            isPotentialPrime = False
            break
        j+=1
    if isPotentialPrime:
        PRIMES.append(i)

HOST = "https://toughsat.appspot.com/generate"
for i in range(len(PRIMES)-1):
    for j in range(i+1,len(PRIMES)):
        factor1=str(PRIMES[i])
        factor2=str(PRIMES[j])
        fileName = TARGET_DIR+'/primes_'+factor1+'_'+factor2+'.dimacs'
        if not os.path.exists(fileName):
            cmd = 'curl -X POST -F "type=factoring2017" -F "factor1='+factor1+'" -F "factor2='+factor2+'" -F "includefactors=on" '+HOST+' > '+fileName
            os.system(cmd)

print(PRIMES)

print("Done.")
