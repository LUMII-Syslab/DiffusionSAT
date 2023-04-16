#!/usr/bin/env python3

import os
import fnmatch
from natsort import natsorted, ns

import random
import hashlib
import subprocess

from data.k_sat import KSAT

# from utils.sat import remove_unused_vars


class SplotData(KSAT):
    """
    Enumerating SAT instances obtained from http://www.splot-research.org (real world use cases).

    The files come in the XML format. They must be placed in <data_dir>.
    """

    def __init__(self, data_dir,
                 test_size_factor,
                 **kwargs) -> None:
        super(SplotData, self).__init__(data_dir, force_data_gen=False, **kwargs)

        self.data_dir = data_dir
        self.test_size_factor = test_size_factor


    def train_generator(self) -> tuple:
        return self.__generator()

    def test_generator(self) -> tuple:
        return self.__generator(True)
    

    def __generator(self, is_test=False) -> tuple:        
        files = natsorted( {x for x in fnmatch.filter(os.listdir(self.data_dir), '*.xml')}, alg=ns.IGNORECASE )
#        total_count = len(files)

        k = int(1/self.test_size_factor) # each k-th file will be a part of the test set; others form the train set

        i = 0
        for fname in files:
            i += 1
            if is_test == (i%k == 0):  # is_test <=> we are a k-th file

                f = open(os.path.join(self.data_dir,fname), 'r')
                lines = f.readlines()

                # count vars and create the map:
                var_map = {}
                clauses = []
                for line in lines:
                    if line.startswith("Clause3CNF_"):
                        clauses.append(self._line2clause(line, var_map))

                yield len(var_map), clauses

            #hex = hashlib.md5(path.name).hexdigest()
            #i = int(hex, 16)
            #print(path.name, i, i % 5)

    def _line2clause(self, line, var_map):
        res = []
        line = line[line.find(":")+1:]
        for id in line.split(" OR "):
            neg = id.startswith("~")
            if neg:
                id = id[1:]
            
            if id not in var_map:
                var_map[id] = len(var_map)+1 # add new variable

            res.append(-var_map[id] if neg else var_map[id])
        return res


if __name__ == "__main__":

    MY_DIR = os.path.dirname(os.path.realpath(__file__))

    # for 
    d = SplotData(os.path.join(MY_DIR,"splot"), 0.1)
    d.__generator(d, True)

    #nvars, clauses = remove_unused_vars(10, [[-1, 4, 9], [-4, 1, -10], [10]])
    # nvars, clauses = remove_unused_vars(3, [[-1,3]])
    # nvars, clauses = remove_unused_vars(3, [[]])
    #print(nvars, clauses)
